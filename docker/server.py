import os
import tempfile
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, PlainTextResponse
from faster_whisper import WhisperModel

MODEL_NAME = os.getenv("FW_MODEL", "large-v3")
DEVICE = os.getenv("FW_DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("FW_COMPUTE_TYPE", "float16")

app = FastAPI()
print(f"Loading model: {MODEL_NAME} on {DEVICE} ({COMPUTE_TYPE})")
model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
print("Model loaded!")

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "device": DEVICE, "compute_type": COMPUTE_TYPE}

@app.get("/status")
def status():
    return {"version": "faster-whisper server, model=" + MODEL_NAME}

@app.post("/asr")
async def asr(
    audio_file: UploadFile = File(...),
    task: str = Form(default="transcribe"),
    language: Optional[str] = Form(default=None),
    output: str = Form(default="srt"),
    encode: str = Form(default="false"),
    video_file: Optional[str] = Form(default=None),
):
    suffix = os.path.splitext(audio_file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        tmp.write(await audio_file.read())
    try:
        segments, info = model.transcribe(
            tmp_path, language=language, task=task, beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 200},
            condition_on_previous_text=False,
            no_speech_threshold=0.6, compression_ratio_threshold=2.4,
            word_timestamps=False,
        )
        seg_list = []
        text_parts = []
        for s in segments:
            seg_list.append({"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text})
            text_parts.append(s.text)
        full_text = "".join(text_parts).strip()
        if output == "json":
            return JSONResponse({"text": full_text})
        elif output == "verbose_json":
            return JSONResponse({"task": task, "language": info.language, "duration": round(seg_list[-1]["end"], 3) if seg_list else 0.0, "text": full_text, "segments": seg_list})
        else:
            srt_lines = []
            for i, s in enumerate(seg_list, 1):
                start_tc = _seconds_to_srt_timecode(s["start"])
                end_tc = _seconds_to_srt_timecode(s["end"])
                srt_lines.append(str(i))
                srt_lines.append(start_tc + " --> " + end_tc)
                srt_lines.append(s["text"].strip())
                srt_lines.append("")
            return PlainTextResponse("\n".join(srt_lines))
    finally:
        try: os.remove(tmp_path)
        except OSError: pass

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
        segments, info = model.transcribe(tmp_path, beam_size=1, vad_filter=True)
        return JSONResponse({"detected_language": info.language, "language_code": info.language, "language_probability": info.language_probability})
    finally:
        try: os.remove(tmp_path)
        except OSError: pass

def _seconds_to_srt_timecode(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return "%02d:%02d:%02d,%03d" % (hours, minutes, secs, millis)