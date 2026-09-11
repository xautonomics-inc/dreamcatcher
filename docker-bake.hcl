variable "REPO_OWNER" { default = "local" }
variable "VARIANT" { default = "cpu" }
variable "BUILD_NUMBER" { default = "0" }
# GIT_SHA: source commit the image was built from. Baked by CI as
# github.sha; on a local checkout it defaults to the current HEAD commit so
# tags are reproducible without extra plumbing.
variable "GIT_SHA" { default = "0000000000000000000000000000000000000000" }
variable "CUDA_VERSION" { default = "12.6.2" }
variable "CUDA_DOCKER_ARCH" { default = "86;90" }
variable "USE_CCACHE" { default = "true" }
variable "GGML_NATIVE" { default = "ON" }

# Registry for all published images. CI overrides; the default is a local
# name so nothing pushes anywhere by accident.
variable "IMAGE_REGISTRY" { default = "ghcr.io" }

# Base-image pins (multi-arch index digests, resolved 2026-09). Keep these
# in sync with the Containerfile ARG defaults; CI passes them explicitly so
# a build never silently follows a moving tag.
#
# NOTE: no `locals`/conditionals here on purpose -- buildx 0.13.1 (Debian)
# rejects a locals block whose members reference variables ("variable cycle
# not allowed for local"). The CUDA base selection is done by the CI matrix
# (which knows its row's CUDA version) or by env override locally, against
# the *variable names below -- HCL variables read same-named env vars.
variable "BASE_UBUNTU" {
  default = "docker.io/library/ubuntu:24.04@sha256:224a1869083a311ef3f13648a154ba79832fbef6364d31493642ca03082da254"
}
# Default = the 12.6.2 pins (matches CUDA_VERSION default and the cuda
# Containerfile ARG defaults). For cu13, override BASE_CUDA_DEV_CONTAINER /
# BASE_CUDA_RUN_CONTAINER (env) with the 13.1.1 digests below.
variable "BASE_CUDA_DEV_CONTAINER" {
  default = "docker.io/nvidia/cuda:12.6.2-devel-ubuntu24.04@sha256:738fba0fbdb225b7a2931c58a5c8f03a84d3cd2f6a84975826a157339ef750b8"
}
variable "BASE_CUDA_RUN_CONTAINER" {
  default = "docker.io/nvidia/cuda:12.6.2-runtime-ubuntu24.04@sha256:16411bb06f363425265b410810a86fc43da705847ee9721be45b478bc4d0ea26"
}
# cu13 pins, for reference/override (13.1.1):
#   devel   @sha256:9cf8694a27722418a1f175d90f85d5afb5a728fd4a9907d7f0565efecfa14d32
#   runtime @sha256:12e26235ebe186000d71f8e457a9ad2aed6c0cb743a7935f0443bacef206aa34
variable "BASE_PYTHON_SLIM" {
  default = "docker.io/library/python:3.12-slim-bookworm@sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254"
}

# Common cache configuration for GitHub Actions
target "cache_settings" {
  cache-from = ["type=gha,scope=ccache-${VARIANT}"]
  cache-to   = ["type=gha,mode=max,scope=ccache-${VARIANT}"]
}

group "default" {
  targets = ["server", "full", "swap"]
}

target "settings" {
  context = "."
  inherits = ["cache_settings"]
  args = {
    BUILD_NUMBER     = "${BUILD_NUMBER}"
    CUDA_VERSION     = "${CUDA_VERSION}"
    CUDA_DOCKER_ARCH = "${CUDA_DOCKER_ARCH}"
    GGML_NATIVE      = "${GGML_NATIVE}"
    USE_CCACHE       = "${USE_CCACHE}"
    BASE_IMAGE       = "${BASE_UBUNTU}"
    BASE_CUDA_DEV_CONTAINER = "${BASE_CUDA_DEV_CONTAINER}"
    BASE_CUDA_RUN_CONTAINER = "${BASE_CUDA_RUN_CONTAINER}"
  }
}

# Tags: the rolling alias ({variant}-{flavor}) AND a source-SHA tag
# ({variant}-{flavor}-{sha7} plus the full sha) so any published alias can
# be traced to an exact commit. The resolved digest goes in the job summary
# via bake's --metadata-file (see .github/workflows/build-container.yml).
target "server" {
  inherits = ["settings"]
  target   = "server"
  tags = [
    "${IMAGE_REGISTRY}/${REPO_OWNER}/ik-llama-cpp:${VARIANT}-server",
    "${IMAGE_REGISTRY}/${REPO_OWNER}/ik-llama-cpp:${VARIANT}-server-${substr(GIT_SHA, 0, 7)}",
    "${IMAGE_REGISTRY}/${REPO_OWNER}/ik-llama-cpp:${VARIANT}-server-${GIT_SHA}",
  ]
}

target "full" {
  inherits = ["settings"]
  target   = "full"
  tags = [
    "${IMAGE_REGISTRY}/${REPO_OWNER}/ik-llama-cpp:${VARIANT}-full",
    "${IMAGE_REGISTRY}/${REPO_OWNER}/ik-llama-cpp:${VARIANT}-full-${substr(GIT_SHA, 0, 7)}",
    "${IMAGE_REGISTRY}/${REPO_OWNER}/ik-llama-cpp:${VARIANT}-full-${GIT_SHA}",
  ]
}

target "swap" {
  inherits = ["settings"]
  target   = "swap"
  tags = [
    "${IMAGE_REGISTRY}/${REPO_OWNER}/ik-llama-cpp:${VARIANT}-swap",
    "${IMAGE_REGISTRY}/${REPO_OWNER}/ik-llama-cpp:${VARIANT}-swap-${substr(GIT_SHA, 0, 7)}",
    "${IMAGE_REGISTRY}/${REPO_OWNER}/ik-llama-cpp:${VARIANT}-swap-${GIT_SHA}",
  ]
}

# Fleet tools image (meta#70): stdlib-only Python tools from the checkout
# (gslot, layer-distribution; fleet-console pending). No CUDA needed, no
# ccache needed -- its own tiny scope.
target "tools" {
  context    = "."
  dockerfile = "./docker/ik_llama-tools.Containerfile"
  target     = "tools"
  args = {
    PYTHON_VERSION = "${BASE_PYTHON_SLIM}"
  }
  tags = [
    "${IMAGE_REGISTRY}/${REPO_OWNER}/ik-llama-cpp:tools",
    "${IMAGE_REGISTRY}/${REPO_OWNER}/ik-llama-cpp:tools-${substr(GIT_SHA, 0, 7)}",
    "${IMAGE_REGISTRY}/${REPO_OWNER}/ik-llama-cpp:tools-${GIT_SHA}",
  ]
}
