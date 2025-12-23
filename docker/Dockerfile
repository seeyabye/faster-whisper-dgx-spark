FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04


ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git cmake ninja-build build-essential pkg-config \
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



# app deps
COPY docker/requirements.txt /app/requirements.txt
RUN python3 -m pip install -r /app/requirements.txt
RUN python3 -m pip uninstall -y ctranslate2 && \
    cd /tmp/ctranslate2/python && \
    python3 -m pip install .

COPY docker/server.py /app/server.py
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/app/entrypoint.sh"]
