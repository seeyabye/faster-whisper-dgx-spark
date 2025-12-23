# faster-whisper on DGX Spark / ThinkStation PGX

This repository provides a minimal, reproducible setup to run `faster-whisper` with CUDA on an NVIDIA DGX Spark–equivalent environment (Lenovo ThinkStation PGX).  
Even if the host already has CUDA 13 installed (and is hard to change), you can run GPU inference reliably by building CTranslate2 with CUDA 12.4 + cuDNN 9 inside Docker.

Official faster-whisper repository: https://github.com/SYSTRAN/faster-whisper

## Why this setup is needed
- `faster-whisper` relies on CTranslate2, and the CUDA/cuDNN versions used at build time must match the runtime environment for the GPU build to work.
- On ThinkStation PGX, CUDA 13 is installed on the host. A simple `pip install ctranslate2` may install a CPU-only or non-CUDA build.
- Following only the official instructions sometimes results in `ValueError: This CTranslate2 package was not compiled with CUDA support`.
- To ensure reproducibility, we explicitly build CTranslate2 from source with CUDA 12.4 + cuDNN 9 in Docker, and reinstall the Python bindings from that build.

## What this repo provides
- `docker-compose.yml` and `docker/Dockerfile` for a GPU-enabled container:
  - Base image: `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`
  - CTranslate2 v4.6.2 built from source with CUDA enabled
  - The CPU/non-CUDA `ctranslate2` wheel installed via pip is uninstalled once, and then replaced with the locally built CUDA-enabled Python bindings
- A minimal FastAPI + Uvicorn server (`docker/server.py`):
  - `POST /transcribe`: receive audio and return Whisper transcription
  - `GET /health`: return model name, device, and compute_type
- Sample audio and test scripts (`test/`)

## Key design points
- CTranslate2 is built with CUDA 12.x + cuDNN 9, and the same version is guaranteed at runtime via the Docker image.
- `entrypoint.sh` explicitly configures `LD_LIBRARY_PATH` and dynamically adds the paths to `nvidia.cublas.lib` and `nvidia.cudnn.lib`.
- Docker is required to isolate from host CUDA 13; GPU access assumes NVIDIA Container Toolkit.

## Prerequisites
- NVIDIA Container Toolkit installed
- GPU driver correctly installed on the host
- Host CUDA can remain 13.x (no need to change)
- Internet connection (for first-time model download)

### Optional: Local CPU-only run (.venv)
If you want to try CPU-only inference without Docker, you can use a local Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r docker/requirements.txt

# Run sample audio on CPU
FW_DEVICE=cpu FW_COMPUTE_TYPE=int8 python3 test/transcribe_cpu.py
```

This path is intended just for quick CPU checks; for GPU we recommend using the Docker setup below.

## Setup (Docker)

```bash
# Build
docker compose build

# Start (GPU required)
docker compose up -d

# Health check
curl http://localhost:8002/health
```

## How to use the inference API

```bash
curl -X POST http://localhost:8002/transcribe \
  -F "file=@test/003.wav" \
  -F "language=ja" \
  -F "beam_size=5" \
  -F "vad_filter=true"
```

The response includes `text`, `segments` (start/end/text), `language`, and `language_probability`.

You can change model and runtime settings via environment variables:
- `FW_MODEL` (default: `large-v3`)
- `FW_DEVICE` (default: `cuda`)
- `FW_COMPUTE_TYPE` (default: `float16`)

### Running `test/` scripts inside the Docker container

If you prefer not to install Python on the host, you can run the test scripts inside the running container:

```bash
# 1) Enter the container (assuming docker compose is up)
docker compose exec whisper bash

# 2) GPU sample (default: large-v3 / float16)
python3 test/transcribe.py

# 3) CPU run (switch compute_type via env vars)
FW_DEVICE=cpu FW_COMPUTE_TYPE=int8 python3 test/transcribe_cpu.py
```

Both scripts use `test/003.wav` as the default input and print the transcript and segments to stdout.  
To test with your own audio file, pass the path as an argument:

```bash
python3 test/transcribe.py /path/to/your.wav
```

## Directory layout (partial)

- `docker-compose.yml` — Service definition (including GPU resources)
- `docker/Dockerfile` — Build CTranslate2 from source on CUDA 12.4 + cuDNN 9
- `docker/entrypoint.sh` — Configure CUDA library paths and start uvicorn
- `docker/server.py` — FastAPI server (`/health`, `/transcribe`)
- `docker/requirements.txt` — Minimal dependencies: `faster-whisper`, `fastapi`, `uvicorn[standard]`, `python-multipart`
- `test/` — Sample audio (`003.wav`) and local test scripts

## Notes for reproducibility

- A simple `pip install faster-whisper` does not guarantee a CUDA-enabled `ctranslate2` build.
- Even if installation succeeds, a CUDA version mismatch may cause the GPU not to be used at runtime.
- This Dockerfile explicitly uninstalls the CPU build once and reinstalls the locally built CUDA-enabled version to keep everything consistent.


