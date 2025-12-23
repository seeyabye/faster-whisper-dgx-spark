import os
import tempfile
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

MODEL_NAME = os.getenv("FW_MODEL", "large-v3")
DEVICE = os.getenv("FW_DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("FW_COMPUTE_TYPE", "float16")

app = FastAPI()

# 起動時にモデルをロードして常駐（ここが重要：毎回ロードしない）
model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "device": DEVICE, "compute_type": COMPUTE_TYPE}

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form(default=None),   # "ja"など
    beam_size: int = Form(default=5),
    vad_filter: bool = Form(default=True),
):
    # アップロードを一時ファイルに保存（PyAVがファイルパスを扱いやすい）
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        tmp.write(await file.read())

    try:
        segments, info = model.transcribe(
            tmp_path,
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
        )

        out_segments = []
        text_all = []
        for s in segments:
            out_segments.append({"start": s.start, "end": s.end, "text": s.text})
            text_all.append(s.text)

        return JSONResponse(
            {
                "language": info.language,
                "language_probability": info.language_probability,
                "text": "".join(text_all).strip(),
                "segments": out_segments,
            }
        )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
