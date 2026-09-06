# syntax=docker/dockerfile:1
#
# sparkinfer + Qwen3.8-27B-NVFP4, one image:
#   serve   AR + image + video over the OpenAI-compatible API   (default)
#   bench   DSpark speculative benchmark                        (`... bench code`)
#
# Built from the source in this repo, so a release tag produces exactly that tag's engine.
# Blackwell (sm_120) only. Weights download themselves on first run into the /models volume.

FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS builder
ARG CUDA_ARCH=120
ENV DEBIAN_FRONTEND=noninteractive
# git: cmake FetchContent clones tokenizers_cpp/cutlass at configure time.
# python3: CUTLASS runs find_package(Python3) at configure time and the CUDA images ship none.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake ninja-build build-essential ca-certificates curl pkg-config python3 \
    && rm -rf /var/lib/apt/lists/*
# tokenizers_cpp needs rustc >= 1.79, and the build resolves cargo at /usr/bin/cargo.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --default-toolchain stable --profile minimal \
    && ln -sf /root/.cargo/bin/cargo /usr/bin/cargo \
    && ln -sf /root/.cargo/bin/rustc /usr/bin/rustc
ENV PATH=/root/.cargo/bin:$PATH
COPY . /src
WORKDIR /src
RUN cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
        -DBUILD_SERVER=ON -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=ON \
 && cmake --build build --target sparkinfer_server qwen38_hf_dflash_bench -j"$(nproc)"
RUN strip build/server/sparkinfer_server build/runtime/qwen38_hf_dflash_bench || true

# base, not runtime: the server links only libcudart (see ldd). The runtime image adds ~3.1 GB
# of cuBLAS/cuSPARSE/cuFFT/cuSOLVER/cuRAND/NPP that nothing here references.
FROM nvidia/cuda:12.8.1-base-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive
# ffmpeg is REQUIRED for video input -- the server shells out to it for frame extraction.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg python3 python3-pip ca-certificates curl \
    && pip3 install --no-cache-dir --break-system-packages "huggingface_hub[cli]" tokenizers jinja2 \
    && apt-get purge -y python3-pip && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /root/.cache

# The server links these two; libcudart comes from the runtime image and libnvidia-ml is
# injected by the NVIDIA container runtime.
COPY --from=builder /usr/local/cuda/lib64/libcudart.so.12*     /opt/sparkinfer/lib/
COPY --from=builder /src/build/server/sparkinfer_server        /opt/sparkinfer/bin/
COPY --from=builder /src/build/runtime/qwen38_hf_dflash_bench  /opt/sparkinfer/bin/
COPY --from=builder /src/build/runtime/libsparkinfer_runtime.so /opt/sparkinfer/lib/
COPY --from=builder /src/build/moe/libsparkinfer_moe.so         /opt/sparkinfer/lib/
COPY docker/entrypoint.sh /opt/sparkinfer/entrypoint.sh
COPY docker/mkids.py      /opt/sparkinfer/mkids.py
RUN echo /opt/sparkinfer/lib > /etc/ld.so.conf.d/sparkinfer.conf && ldconfig \
 && chmod +x /opt/sparkinfer/entrypoint.sh /opt/sparkinfer/bin/*

ENV MODEL_REPO=gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090 \
    MODEL_DIR=/models/qwen38-nvfp4 \
    DRAFT_REPO=gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4 \
    DRAFT_DIR=/models/qwen38-dspark \
    MODEL_NAME=qwen38-nvfp4 \
    CTX=262144 HOST=0.0.0.0 PORT=8080 \
    SPARKINFER_MAX_OUTPUT_TOKENS=16384 \
    HF_HUB_DISABLE_TELEMETRY=1
VOLUME ["/models"]
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=30m --retries=3 \
  CMD curl -fsS "http://localhost:${PORT}/health" || exit 1
ENTRYPOINT ["/opt/sparkinfer/entrypoint.sh"]
