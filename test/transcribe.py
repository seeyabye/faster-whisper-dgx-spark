#!/usr/bin/env python3
# test/transcribe_test.py

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from faster_whisper import WhisperModel


# === ここを自分のwavに合わせて書き換え ===
WAV_PATH = Path(__file__).resolve().parent / "003.wav"
MODEL_NAME = "large-v3"
LANGUAGE = "ja"
BEAM_SIZE = 5
COMPUTE_TYPE = "float16"   # GPU想定
# =======================================


def pick_device():
    return "cuda", COMPUTE_TYPE


def main() -> int:
    wav_path = WAV_PATH
    if len(sys.argv) >= 2:
        wav_path = Path(sys.argv[1]).expanduser().resolve()

    if not wav_path.exists():
        print(f"[ERROR] wav not found: {wav_path}", file=sys.stderr)
        return 2

    device, compute_type = pick_device()
    print(f"[INFO] model={MODEL_NAME} device={device} compute_type={compute_type}")
    print(f"[INFO] input={wav_path}")

    model = WhisperModel(
        MODEL_NAME,
        device=device,
        compute_type=compute_type,
    )

    # ==== 計測開始 ====
    t0 = time.perf_counter()

    segments, info = model.transcribe(
        str(wav_path),
        language=LANGUAGE,
        beam_size=BEAM_SIZE,
        # vad_filter=True,
    )

    segments = list(segments)  # ここで実際に推論が走る

    # ==== 計測終了 ====
    t1 = time.perf_counter()
    elapsed = t1 - t0

    print(f"\n[INFO] detected_language={info.language} (p={info.language_probability:.3f})")
    print(f"[INFO] transcription_time={elapsed:.3f} sec")
    print("[RESULT] --- transcript ---")
    for seg in segments:
        print(f"[{seg.start:7.2f}-{seg.end:7.2f}] {seg.text}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())