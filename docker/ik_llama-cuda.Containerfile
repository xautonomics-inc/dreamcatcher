ARG CUDA_VERSION=12.6.2
# Base images pinned per CUDA variant (docker.io/nvidia/cuda, multi-arch
# index digests resolved 2026-09). Bake pins both the tag and the digest from
# docker-bake.hcl; defaults here keep a standalone `docker build` working.
ARG BASE_CUDA_DEV_CONTAINER=docker.io/nvidia/cuda:12.6.2-devel-ubuntu24.04@sha256:738fba0fbdb225b7a2931c58a5c8f03a84d3cd2f6a84975826a157339ef750b8
ARG BASE_CUDA_RUN_CONTAINER=docker.io/nvidia/cuda:12.6.2-runtime-ubuntu24.04@sha256:16411bb06f363425265b410810a86fc43da705847ee9721be45b478bc4d0ea26

# Stage 1: Build
FROM ${BASE_CUDA_DEV_CONTAINER} AS build

# Build arguments
ARG CUDA_DOCKER_ARCH="75-virtual;80-virtual;86-real;89-real"
ARG GGML_NATIVE=ON
ARG USE_CCACHE=true

# Environment variables for portability and GitHub Actions
ENV CCACHE_DIR=/ccache
ENV CCACHE_UMASK=000
ENV CCACHE_MAXSIZE=5G
ENV CCACHE_COMPRESS=1
ENV CCACHE_BASEDIR=/app

RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
    ca-certificates build-essential libcurl4-openssl-dev curl libgomp1 cmake ccache git libibverbs-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy non-hidden files first
COPY . /app

WORKDIR /app

# Build using ccache and optional custom commit
RUN --mount=type=cache,target=/ccache \
    --mount=type=bind,source=.git,target=.git \
    if [ "${USE_CCACHE}" = "true" ]; then \
        export PATH="/usr/lib/ccache:$PATH"; \
        ccache -z; \
    fi && \
    cmake -B build \
        -DGGML_NATIVE=${GGML_NATIVE} \
        -DGGML_CUDA=ON \
        -DGGML_RPC=ON -DGGML_RPC_RDMA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_DOCKER_ARCH}" \
        -DLLAMA_CURL=ON \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined && \
    cmake --build build --config Release -j$(nproc) && \
    if [ "${USE_CCACHE}" = "true" ]; then \
        ccache -s; \
    fi

# Collect build artifacts
RUN mkdir -p /app/dist/lib /app/dist/full /app/dist/bin && \
    find build -name "*.so" -exec cp {} /app/dist/lib \; && \
    cp build/bin/* /app/dist/bin/ && \
    cp build/bin/* /app/dist/full/ && \
    cp *.py /app/dist/full/ && \
    cp -r gguf-py /app/dist/full/ && \
    cp -r requirements /app/dist/full/ && \
    cp requirements.txt /app/dist/full/ && \
    cp .devops/tools.sh /app/dist/full/

# Server-stage payload. llama-expert-server is not a build target yet (tracked
# gap), so it is collected only when present; the rest always ships.
RUN mkdir -p /app/dist/server && \
    cp /app/dist/bin/llama-server /app/dist/bin/llama-stage-runner /app/dist/server/ && \
    if [ -f /app/dist/bin/llama-expert-server ]; then \
        cp /app/dist/bin/llama-expert-server /app/dist/server/; \
    fi

# Stage 2: Base (Shared Runtime)
FROM ${BASE_CUDA_RUN_CONTAINER} AS base
# rdma-core: libibverbs runtime for the RoCE transport tier. The transport
# probes for a device at startup and falls back to plain TCP when
# /dev/infiniband is absent, so this costs nothing on non-RDMA hosts.
RUN apt-get update && \
    apt-get install -yq --no-install-recommends libgomp1 curl ca-certificates rdma-core libibverbs1 && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV LD_LIBRARY_PATH=/app/lib
COPY --from=build /app/dist/lib /app/lib

# Stage 3: Full (Python/Dev Tools)
FROM base AS full
COPY --from=build /app/dist/full /app
RUN apt-get update && \
    apt-get install -yq --no-install-recommends git python3 python3-pip && \
    pip install --break-system-packages -r requirements.txt && \
    rm -rf /var/lib/apt/lists/*
ENTRYPOINT ["/app/tools.sh"]

# Stage 4: Server (llama-server + multi-stage drivers)
FROM base AS server
ENV LLAMA_ARG_HOST=0.0.0.0
# Multi-stage split serving: the runner drives stage-server roles over the
# hidden-state transport (TCP, auto-negotiating RDMA when both ends see RoCE).
# llama-expert-server lands here automatically when that target exists (see
# the collect step -- it is not a build target yet, tracked gap).
COPY --from=build /app/dist/server/ /app/
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD [ "curl", "-f", "http://localhost:8080/health" ]
ENTRYPOINT [ "/app/llama-server" ]

# Stage 5: Swap
FROM server AS swap
ARG LS_REPO=mostlygeek/llama-swap
ARG LS_VER=239
RUN curl -sSL "https://github.com/${LS_REPO}/releases/download/v${LS_VER}/llama-swap_${LS_VER}_linux_amd64.tar.gz" \
    | tar -xz

COPY --from=build /app/docker/ik_llama-cuda-swap.config.yaml /app/config.yaml
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD [ "curl", "-f", "http://localhost:8080"]
ENTRYPOINT [ "/app/llama-swap", "-config", "/app/config.yaml" ]
