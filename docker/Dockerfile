FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04


ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git cmake ninja-build build-essential pkg-config \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# python basics
RUN python3 -m pip install -U pip setuptools wheel

# Build CTranslate2 from source with CUDA ON (and submodules!)
ARG CT2_VERSION=v4.6.2
RUN git clone --depth 1 --branch v4.6.2 --recurse-submodules \
    https://github.com/OpenNMT/CTranslate2.git /tmp/ctranslate2 && \
    cd /tmp/ctranslate2 && \
    mkdir build && cd build && \
    cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DWITH_CUDA=ON \
      -DWITH_CUDNN=ON \
      -DWITH_MKL=OFF \
      -DOPENMP_RUNTIME=COMP \
      -DCMAKE_CUDA_ARCHITECTURES=native \
      -GNinja && \
    ninja -j$(nproc) && \
    ninja install && \
    ldconfig

# Install CUDA torch from the cu130 index BEFORE whisperx, so whisperx's
# transitive torch dependency is already satisfied and won't be downgraded
# to a CPU wheel from PyPI.
RUN python3 -m pip install --index-url https://download.pytorch.org/whl/cu130 \
    torch==2.10.0+cu130 \
    torchaudio==2.10.0+cu130 \
    torchvision==0.25.0+cu130

# Install whisperx without deps (torch is already installed above), then
# install whisperx's other runtime deps that won't pull a CPU torch.
RUN python3 -m pip install --no-deps whisperx==3.8.6 && \
    python3 -m pip install \
      ctranslate2 huggingface-hub nltk omegaconf pandas \
      pyannote-audio transformers

# Reinstall the source-built CTranslate2 over any pip wheel
RUN python3 -m pip uninstall -y ctranslate2 && \
    cd /tmp/ctranslate2/python && \
    python3 -m pip install .

# app deps
COPY docker/requirements.txt /app/requirements.txt
RUN python3 -m pip install -r /app/requirements.txt

COPY docker/server.py /app/server.py
COPY docker/timing.py /app/timing.py
COPY docker/parakeet.py /app/parakeet.py
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/app/entrypoint.sh"]