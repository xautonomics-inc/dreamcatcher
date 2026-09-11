# Base image pinned by digest (docker.io/library/ubuntu 24.04, multi-arch
# index; resolved 2026-09). Bake passes BASE_IMAGE explicitly; override here
# with a fresh digest when refreshing.
ARG BASE_IMAGE=docker.io/library/ubuntu:24.04@sha256:224a1869083a311ef3f13648a154ba79832fbef6364d31493642ca03082da254

# Stage 1: Build
FROM ${BASE_IMAGE} AS build

# Build arguments
ARG GGML_NATIVE=ON
ARG GGML_AVX2=ON
ARG USE_CCACHE=true

# Environment variables for portability and GitHub Actions
ENV LLAMA_CURL=1
ENV LC_ALL=C.utf8

# ccache configuration
ENV CCACHE_DIR=/ccache
ENV CCACHE_MAXSIZE=1G
ENV CCACHE_COMPRESS=1
ENV CCACHE_COMPRESSLEVEL=6
# This is CRITICAL for GitHub Actions: it ignores the absolute path of the runner
ENV CCACHE_BASEDIR=/app

RUN apt-get update && \
    apt-get install -yq --no-install-recommends ca-certificates build-essential libcurl4-openssl-dev curl libgomp1 cmake ccache git libibverbs-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy source code (excluding hidden files/dirs via .dockerignore)
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
        -DLLAMA_CURL=ON \
        -DGGML_RPC=ON -DGGML_RPC_RDMA=ON && \
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
ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS base
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

COPY --from=build /app/docker/ik_llama-cpu-swap.config.yaml /app/config.yaml
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD [ "curl", "-f", "http://localhost:8080"]
ENTRYPOINT [ "/app/llama-swap", "-config", "/app/config.yaml" ]
