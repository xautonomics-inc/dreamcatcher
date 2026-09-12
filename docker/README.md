# Build and use ik_llama.cpp with CPU or CPU+CUDA

Built on top of [ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) and [llama-swap](https://github.com/mostlygeek/llama-swap)

Commands are provided for Podman and Docker.

CPU or CUDA sections under [Prebuilt](#Prebuilt)/[Build](#Build) and [Run]($Run) are enough to get up and running.

## Overview

- [Prebuilt](#Prebuilt)
- [Build](#Build)
- [Run](#Run)
- [Troubleshooting](#Troubleshooting)
- [Extra Features](#Extra)
- [Credits](#Credits)

## Prebuilt Docker images

Pull one of the available images from `ghcr.io`. [View all tags](https://github.com/ikawrakow/ik_llama.cpp/pkgs/container/ik-llama-cpp/versions?filters%5Bversion_type%5D=tagged)

```bash
docker pull ghcr.io/ikawrakow/ik-llama-cpp:cpu-swap
docker pull ghcr.io/ikawrakow/ik-llama-cpp:cpu-server
docker pull ghcr.io/ikawrakow/ik-llama-cpp:cpu-full

docker pull ghcr.io/ikawrakow/ik-llama-cpp:cu12-swap
docker pull ghcr.io/ikawrakow/ik-llama-cpp:cu12-server
docker pull ghcr.io/ikawrakow/ik-llama-cpp:cu12-full
```

## Build

The project uses Docker Bake for building multiple targets efficiently.

Clone the repository: `git clone https://github.com/ikawrakow/ik_llama.cpp`

Use `docker-bake`.

```bash
docker buildx create --name ik-llama-builder --use
```

### CPU Variant

```bash
VARIANT=cpu docker buildx bake --builder ik-llama-builder --load full swap
```

The `server` target (also ships `llama-stage-runner`) needs its Containerfile
set explicitly outside the default group, which the CI workflow does with a
`--set`. Locally:

```bash
VARIANT=cpu docker buildx bake --builder ik-llama-builder --load \
  --set server.dockerfile=docker/ik_llama-cpu.Containerfile server
```

Or with custom tags:

```bash
REPO_OWNER=yourname VARIANT=cpu docker buildx bake --builder ik-llama-builder --load \
  -f ./docker-bake.hcl \
  full swap
```

### CUDA Variant

First, set the CUDA version and GPU architecture in `ik_llama-cuda.Containerfile`:
- `CUDA_DOCKER_ARCH`: Your GPU's compute capability (e.g., `86` for RTX 30*, `89` for RTX 40*, `12.0` for RTX 50*)
- `CUDA_VERSION`: CUDA Toolkit version (e.g., `12.6.2`, `13.1.1`)

```bash
VARIANT=cu12 docker buildx bake --builder ik-llama-builder --load full swap
```

Base images are pinned by digest. The bake file defaults to the **12.6.2**
pins; for a cu13 build pass the matching pins (see the `cu13` matrix row in
`.github/workflows/build-container.yml` for current digests):

```bash
VARIANT=cu13 \
BASE_CUDA_DEV_CONTAINER=docker.io/nvidia/cuda:13.1.1-devel-ubuntu24.04@sha256:9cf8694a27722418a1f175d90f85d5afb5a728fd4a9907d7f0565efecfa14d32 \
BASE_CUDA_RUN_CONTAINER=docker.io/nvidia/cuda:13.1.1-runtime-ubuntu24.04@sha256:12e26235ebe186000d71f8e457a9ad2aed6c0cb743a7935f0443bacef206aa34 \
docker buildx bake --builder ik-llama-builder --load full swap
```

The bake file deliberately avoids `locals`/conditionals so it parses on
distro-buildx (tested against 0.13.1); base selection is env/variable-
driven.

### Build Targets

Builds two image tags per variant:

- **`full`**: Includes `llama-server`, `llama-quantize`, and other utilities.
- **`swap`**: Includes only `llama-swap` and `llama-server`.

CI additionally tags every image with the source SHA (`…-<sha>`). Locally that
tag comes from the bake **variable** `GIT_SHA`, not a build arg — export it or
a local bake tags `…-0000000`:

```bash
GIT_SHA=$(git rev-parse --short HEAD) VARIANT=cpu docker buildx bake …
```

## Run

- Download `.gguf` model files to your favorite directory (e.g., `/my_local_files/gguf`).
- Map it to `/models` inside the container.
- Open browser `http://localhost:9292` and enjoy the features.
- API endpoints are available at `http://localhost:9292/v1` for use in other applications.

### CPU

```bash
podman run -it --name ik_llama --rm -p 9292:8080 -v /my_local_files/gguf:/models:ro localhost/ik_llama-cpu:swap
```

```bash
docker run -it --name ik_llama --rm -p 9292:8080 -v /my_local_files/gguf:/models:ro localhost/ik_llama-cpu:swap
```

### CUDA

- Install Nvidia Drivers and CUDA on the host.
- For Docker, install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- For Podman, install [CDI Container Device Interface](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html)
- Identify your GPU:
  - [CUDA GPU Compute Capability](https://developer.nvidia.com/cuda/gpus) (e.g., `8.6` for RTX30*, `8.9` for RTX40*, `12.0` for RTX50*)
  - [CUDA Toolkit supported version](https://developer.nvidia.com/cuda-toolkit-archive)

```bash
podman run -it --name ik_llama --rm -p 9292:8080 -v /my_local_files/gguf:/models:ro --device nvidia.com/gpu=all --security-opt=label=disable localhost/ik_llama-cuda:swap
```

```bash
docker run -it --name ik_llama --rm -p 9292:8080 -v /my_local_files/gguf:/models:ro --runtime nvidia localhost/ik_llama-cuda:swap
```

## Troubleshooting

- If CUDA is not available, use `ik_llama-cpu` instead.
- If models are not found, ensure you mount the correct directory: `-v /my_local_files/gguf:/models:ro`
- If you need to install `podman` or `docker` follow the [Podman Installation](https://podman.io/docs/installation) or [Install Docker Engine](https://docs.docker.com/engine/install) for your OS.

## Extra

- **Custom commit**: Build a specific `ik_llama.cpp` commit by modifying the Containerfile or using build args.

```bash
docker buildx bake --builder ik-llama-builder --set full.args.BUILD_COMMIT=1ec12b8 full
```

- **Using the tools in the `full` image**:

```bash
$ podman run -it --name ik_llama_full --rm -v /my_local_files/gguf:/models:ro --entrypoint bash localhost/ik_llama-cpu:full
# ./llama-quantize ...
# python3 gguf-py/scripts/gguf_dump.py ...
# ./llama-perplexity ...
# ./llama-sweep-bench ...
```

```bash
docker run -it --name ik_llama_full --rm -v /my_local_files/gguf:/models:ro --runtime nvidia --entrypoint bash localhost/ik_llama-cuda:full
# ./llama-quantize ...
# python3 gguf-py/scripts/gguf_dump.py ...
# ./llama-perplexity ...
# ./llama-sweep-bench ...
```

- **Customize `llama-swap` config**: Save the `./docker/ik_llama-cpu-swap.config.yaml` or `./docker/ik_llama-cuda-swap.config.yaml` locally (e.g., under `/my_local_files/`) then map it to `/app/config.yaml` inside the container appending `-v /my_local_files/ik_llama-cpu-swap.config.yaml:/app/config.yaml:ro` to your `podman run ...` or `docker run ...`.

- **Run in background**: Replace `-it` with `-d`: `podman run -d ...` or `docker run -d ...`. To stop it: `podman stop ik_llama` or `docker stop ik_llama`.

- **GGML_NATIVE**: If you build the image on a different machine, change `-DGGML_NATIVE=ON` to `-DGGML_NATIVE=OFF` in the `.Containerfile`.

- **KV quantization types**: To use more KV quantization types, build with `-DGGML_IQK_FA_ALL_QUANTS=ON`.

- **Cleanup unused CUDA images**: If you experiment with several `CUDA_VERSION`, delete unused images (they are several GB):
  ```bash
  podman image rm docker.io/nvidia/cuda:12.4.0-runtime-ubuntu22.04 && \
    podman image rm docker.io/nvidia/cuda:12.4.0-devel-ubuntu22.04
  ```

- **Build without `llama-swap`**: Change `--target swap` to `--target server` in docker-bake or Containerfiles.

- **Pre-made quants**: Look for premade quants from [ubergarm](https://huggingface.co/ubergarm/models).

- **GGUF tools**: Build custom quants with [Thireus](https://github.com/Thireus/GGUF-Tool-Suite)'s tools.

- **Download prebuilt binaries**: Download from [ik_llama.cpp's Thireus fork with release builds for macOS/Windows/Ubuntu CPU and Windows CUDA](https://github.com/Thireus/ik_llama.cpp).

- **KoboldCPP experience**: [Croco.Cpp is a fork of KoboldCPP inferring GGUF/GGML models on CPU/Cuda with KoboldAI's UI. It's powered partly by IK_LLama.cpp, and compatible with most of Ikawrakow's quants except Bitnet.](https://github.com/Nexesenex/croco.cpp)

## Credits

All credits to the awesome community:

[llama-swap](https://github.com/mostlygeek/llama-swap)

## Expert-parallel staging (`llama-stage-runner`) and the tools image

The `server`/`swap` images ship `llama-stage-runner` beside `llama-server`: a
thin driver that loads one stage of a very deep MoE (graph-build layer window
via `STAGE_IL_START`/`STAGE_IL_END`), streams activations to the next stage
over TCP **or RDMA**, and can serve the head stage's OpenAI HTTP endpoint.
`llama-expert-server` is **not present in this fork's tree** and therefore not
in the images; porting it is tracked separately.

**Serve a chat model from the `server` image** — note `--jinja`: several chat
templates (e.g. gemma) are rejected without it and requests to `/v1/chat/…`
answer 500:

```bash
docker run --rm -p 9292:8080 -v /my_local_files/gguf:/models:ro \
  localhost/ik_llama-cpu:server -m /models/model.gguf --jinja -c 2048 -t 8
```

**RDMA (RoCE) transport.** Servers are compiled with `GGML_RPC=ON` and
`GGML_RPC_RDMA=ON` (libibverbs). Stage handshakes exchange transport
capabilities and use RDMA when **both** ends expose `/dev/infiniband`; with
the device absent on either side they fall back to TCP, so the same image
runs unchanged on non-RDMA hosts. To use it, pass the device through:

```bash
docker run --rm --device /dev/infiniband ik_llama-cpu:server ...
```

Stage-runner roles (`server` = head with HTTP API, `head`, `tail`, `relay`)
and the full knob list (`--slots`, `--connect`, `--listen`, `--return-listen`,
`--n-ctx`, `--tensor-split`, `-ot`, `-cmoe`, ...) are in
`examples/stage-runner/stage-runner.cpp`. KV geometry must be byte-identical
across stages — start from `gslot plan` (`tools/gslot/README.md`) rather than
hand-rolling a launch set.

**Tools image.** `ik-llama-cpp:tools` is a minimal Python 3.12 image with the
`gslot` placement planner and the layer-distribution analysis scripts frozen
into it (stdlib-only, no network). It is built and tagged by the bake matrix
but **not yet published** — no registry publish path is wired for it yet
(tracked gap). `fleet-console` itself is not in this repository yet; its
planned image path is commented in
`docker/ik_llama-tools.Containerfile`.
