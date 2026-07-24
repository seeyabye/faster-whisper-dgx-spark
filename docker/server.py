import os
import tempfile
import json
import re
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from faster_whisper import WhisperModel
import whisperx

from timing import (
    seconds_to_srt_timecode,
    faster_whisper_words_to_stable_ts,
    adapt_whisperx_words,
    build_result_from_whisperx,
    build_result_from_faster_whisper,
)

MODEL_NAME = os.getenv("FW_MODEL", "large-v3")
DEVICE = os.getenv("FW_DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("FW_COMPUTE_TYPE", "float16")

app = FastAPI()
print(f"Loading model: {MODEL_NAME} on {DEVICE} ({COMPUTE_TYPE})")
model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
print("Model loaded!")

print("Loading WhisperX alignment model (GPU)...")
align_model, align_metadata = whisperx.load_align_model(
    language_code="en",
    device=DEVICE,
    model_name="WAV2VEC2_ASR_BASE_960H",
)
print("WhisperX alignment model loaded on GPU!")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "device": DEVICE, "compute_type": COMPUTE_TYPE}


@app.get("/status")
def status():
    return {"version": "faster-whisper + whisperx, model=" + MODEL_NAME}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": "whisper-large-v3", "object": "model", "created": 0, "owned_by": "faster-whisper"}],
    }


@app.post("/asr")
async def asr(
    audio_file: UploadFile = File(...),
    task: str = Form(default="transcribe"),
    language: Optional[str] = Form(default=None),
    output: str = Form(default="srt"),
    encode: str = Form(default="false"),
    video_file: Optional[str] = Form(default=None),
):
    return await _do_asr(audio_file, task, language, output)


@app.post("/v1/audio/transcriptions")
async def openai_transcribe(
    file: UploadFile = File(...),
    model: str = Form(default="whisper-large-v3"),
    language: Optional[str] = Form(default=None),
    task: str = Form(default="transcribe"),
    response_format: str = Form(default="json"),
    # OpenAI sends timestamp_granularities[] as repeated multipart fields.
    # FastAPI binds the literal field name, so alias to the bracketed form.
    # NOTE: this server always returns words NESTED inside segments[*].words
    # (the stable-ts contract), on both /asr and /v1/audio/transcriptions.
    # OpenAI's top-level words array is NOT supported. Clients needing that
    # shape require a different server.
    timestamp_granularities: Optional[list[str]] = Form(default=None, alias="timestamp_granularities[]"),
):
    if response_format == "verbose_json":
        output = "verbose_json"
    elif response_format in ("json", "text"):
        output = "json"
    else:
        output = "srt"
    return await _do_asr(file, task, language, output)


@app.post("/detect-language")
async def detect_language(
    audio_file: UploadFile = File(...),
    encode: str = Form(default="false"),
    video_file: Optional[str] = Form(default=None),
):
    suffix = os.path.splitext(audio_file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        tmp.write(await audio_file.read())
    try:
        segments, info = model.transcribe(
            tmp_path,
            beam_size=1,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 200,
            },
        )
        return {
            "detected_language": info.language,
            "language_probability": round(info.language_probability, 3),
        }
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


async def _do_asr(audio_file, task, language, output):
    suffix = os.path.splitext(audio_file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        tmp.write(await audio_file.read())
    try:
        # Step 1: Transcribe with faster-whisper.
        # word_timestamps=True so the fallback path (if WhisperX fails or
        # produces unusable words) still has words for stable-ts.
        # large-v3 capitalization is not affected by this flag.
        segments, info = model.transcribe(
            tmp_path,
            language=language,
            task=task,
            beam_size=5,
            vad_filter=True,
            word_timestamps=True,
            vad_parameters={
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 200,
            },
        )

        seg_list = []
        text_parts = []
        for s in segments:
            seg_list.append({
                "start": s.start,
                "end": s.end,
                "text": s.text.strip(),
                "raw_words": s.words,
            })
            text_parts.append(s.text.strip())

        full_text = " ".join(text_parts).strip()
        lang = language or info.language

        # Step 2: Align with WhisperX. Keep its words and its per-segment text.
        # If alignment fails OR any text-bearing segment has no usable words,
        # fall back the WHOLE request to faster-whisper words (no mixing).
        result_segments = []
        try:
            aligned_segments = whisperx.align(
                seg_list,
                align_model,
                align_metadata,
                tmp_path,
                DEVICE,
                return_char_alignments=False,
            )
            result_segments, ok = build_result_from_whisperx(aligned_segments)
            if not ok or not result_segments:
                print("[whisperx] incomplete words; falling back to faster-whisper words")
                result_segments = build_result_from_faster_whisper(seg_list)
        except Exception as e:
            print(f"[whisperx] Alignment failed: {e}, using faster-whisper words")
            result_segments = build_result_from_faster_whisper(seg_list)

        if output == "json":
            return JSONResponse({"text": full_text})
        elif output == "verbose_json":
            return JSONResponse({
                "task": task,
                "language": lang,
                "duration": round(result_segments[-1]["end"], 3) if result_segments else 0.0,
                "text": " ".join(s["text"] for s in result_segments),
                "segments": result_segments,
            })
        else:
            srt_lines = []
            for i, s in enumerate(result_segments, 1):
                start_tc = seconds_to_srt_timecode(s["start"])
                end_tc = seconds_to_srt_timecode(s["end"])
                srt_lines.append(str(i))
                srt_lines.append(start_tc + " --> " + end_tc)
                srt_lines.append(s["text"])
                srt_lines.append("")
            return PlainTextResponse("\n".join(srt_lines))
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass