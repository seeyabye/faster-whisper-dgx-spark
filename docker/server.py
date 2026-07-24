import os
import tempfile
import json
import re
import logging
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

logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("FW_MODEL", "large-v3")
FW_MODELS = os.getenv("FW_MODELS", "")  # comma-separated, e.g. "large-v3,large-v3-turbo"
DEVICE = os.getenv("FW_DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("FW_COMPUTE_TYPE", "float16")
PARAKEET_MODEL = os.getenv("FW_PARAKEET_MODEL", "")  # e.g. "nvidia/parakeet-tdt-0.6b-v3"

app = FastAPI()

# --- Model registry --- #
# Each model is loaded once at startup and registered by its canonical ID.
# Routing uses exact model ID match (no aliases that could serve wrong model).

# faster-whisper models: load from FW_MODELS (comma-separated) or FW_MODEL (single)
fw_model_names = [m.strip() for m in FW_MODELS.split(",") if m.strip()] if FW_MODELS else [MODEL_NAME]
fw_models = {}  # {model_name: WhisperModel}
for name in fw_model_names:
    print(f"Loading faster-whisper model: {name} on {DEVICE} ({COMPUTE_TYPE})")
    fw_models[name] = WhisperModel(name, device=DEVICE, compute_type=COMPUTE_TYPE)
    print(f"faster-whisper model '{name}' loaded!")
# Default model (first in the list, or MODEL_NAME)
fw_default = fw_model_names[0]

# WhisperX alignment model (for faster-whisper path only)
print("Loading WhisperX alignment model (GPU)...")
align_model, align_metadata = whisperx.load_align_model(
    language_code="en",
    device=DEVICE,
    model_name="WAV2VEC2_ASR_BASE_960H",
)
print("WhisperX alignment model loaded on GPU!")

# Parakeet backend (optional, loaded if FW_PARAKEET_MODEL is set)
parakeet_backend = None
if PARAKEET_MODEL:
    try:
        from parakeet import ParakeetBackend
        print(f"Loading Parakeet model: {PARAKEET_MODEL}")
        parakeet_backend = ParakeetBackend(model_id=PARAKEET_MODEL, device=DEVICE)
        print("Parakeet model loaded!")
    except Exception as e:
        print(f"Failed to load Parakeet model: {e}")

# Parakeet routing: exact match on the canonical model ID
PARAKEET_MODEL_IDS = {PARAKEET_MODEL.lower(), "parakeet-tdt-0.6b-v3", "parakeet"} if PARAKEET_MODEL else set()


def get_backend(model_name):
    """Route to the correct backend by model name.

    Returns (backend_type, model_obj) where backend_type is
    'parakeet' or 'faster-whisper'. Falls back to fw_default
    if model_name is None or not found.
    """
    if model_name and parakeet_backend and model_name.lower() in PARAKEET_MODEL_IDS:
        return ("parakeet", parakeet_backend)
    if model_name and model_name in fw_models:
        return ("faster-whisper", fw_models[model_name])
    # Fallback to default
    if model_name:
        print(f"[routing] unknown model={model_name}, using default: {fw_default}")
    return ("faster-whisper", fw_models[fw_default])


@app.get("/health")
def health():
    models = list(fw_models.keys())
    if parakeet_backend:
        models.append(PARAKEET_MODEL)
    return {"status": "ok", "models": models, "device": DEVICE, "compute_type": COMPUTE_TYPE}


@app.get("/status")
def status():
    version = "faster-whisper + whisperx"
    if parakeet_backend:
        version += " + parakeet"
    return {"version": version + ", models=" + ",".join(fw_models.keys())}


@app.get("/v1/models")
def list_models():
    data = [{"id": name, "object": "model", "created": 0, "owned_by": "faster-whisper"} for name in fw_models]
    if parakeet_backend:
        data.append({"id": PARAKEET_MODEL, "object": "model", "created": 0, "owned_by": "nvidia"})
    return {"object": "list", "data": data}


@app.post("/asr")
async def asr(
    audio_file: UploadFile = File(...),
    task: str = Form(default="transcribe"),
    language: Optional[str] = Form(default=None),
    output: str = Form(default="srt"),
    encode: str = Form(default="false"),
    video_file: Optional[str] = Form(default=None),
    model: Optional[str] = Form(default=None, description="Model to use (routing)"),
    skip_align: bool = Query(default=False, description="Skip WhisperX alignment (A/B test)"),
):
    return await _do_asr(audio_file, task, language, output, model, skip_align)


@app.post("/v1/audio/transcriptions")
async def openai_transcribe(
    file: UploadFile = File(...),
    model: str = Form(default="whisper-large-v3"),
    language: Optional[str] = Form(default=None),
    task: str = Form(default="transcribe"),
    response_format: str = Form(default="json"),
    timestamp_granularities: Optional[list[str]] = Form(default=None, alias="timestamp_granularities[]"),
    skip_align: bool = Query(default=False, description="Skip WhisperX alignment (A/B test)"),
):
    if response_format == "verbose_json":
        output = "verbose_json"
    elif response_format in ("json", "text"):
        output = "json"
    else:
        output = "srt"
    return await _do_asr(file, task, language, output, model, skip_align)


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
        segments, info = fw_models[fw_default].transcribe(
            tmp_path,
            beam_size=1,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 200},
        )
        return {"detected_language": info.language, "language_probability": round(info.language_probability, 3)}
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


async def _do_asr(audio_file, task, language, output, model_name=None, skip_align=False):
    suffix = os.path.splitext(audio_file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        tmp.write(await audio_file.read())
    try:
        # Route to the correct backend
        backend_type, backend = get_backend(model_name)
        if backend_type == "parakeet":
            print(f"[routing] using parakeet backend for model={model_name}")
            segments, full_text = backend.transcribe(
                tmp_path, language=language, task=task
            )
            # Parakeet produces native word timestamps — no WhisperX alignment needed.
            result_segments = segments
            lang = language or "en"
        else:
            # faster-whisper path
            segments, info = backend.transcribe(
                tmp_path,
                language=language,
                task=task,
                beam_size=5,
                vad_filter=True,
                word_timestamps=True,
                vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 200},
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

            # WhisperX alignment (unless skip_align)
            result_segments = []
            if skip_align:
                print("[ab] skip_align=True, using faster-whisper words only")
                result_segments = build_result_from_faster_whisper(seg_list)
            else:
                try:
                    aligned_segments = whisperx.align(
                        seg_list, align_model, align_metadata,
                        tmp_path, DEVICE, return_char_alignments=False,
                    )
                    result_segments, ok, synth_count = build_result_from_whisperx(aligned_segments)
                    if not ok or not result_segments:
                        print("[whisperx] text reconstruction failed; falling back to faster-whisper words")
                        result_segments = build_result_from_faster_whisper(seg_list)
                    elif synth_count:
                        print(f"[whisperx] {synth_count}/{len(result_segments)} segments used synthetic words (alignment failed)")
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
                "text": full_text if isinstance(full_text, str) else " ".join(result_segments),
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