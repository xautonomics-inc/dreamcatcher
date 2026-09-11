# Fleet tools image (meta#70): the Python tools frozen from the checkout --
# gslot (compute-slot arbiter) and layer-distribution (GGUF layer-library
# engine). Both are stdlib-only (gslot declares dependencies = [] and
# layer_distribution imports nothing outside the stdlib), so installing them
# pulls zero third-party packages: `pip install --no-deps` on local sources
# only, no network resolution, no pip at runtime.
#
# fleet-console is not in this repo yet (tracked gap, meta#70); when it lands
# under tools/ with a pyproject.toml, add one install line below.
#
# Base pinned by digest (docker.io/library/python 3.12-slim-bookworm,
# multi-arch index; resolved 2026-09).
ARG PYTHON_VERSION=3.12-slim-bookworm@sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254

FROM ${PYTHON_VERSION} AS tools

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /opt/tools

# Build-context paths: this Containerfile is built with the repo root as the
# context (`docker buildx build -f docker/ik_llama-tools.Containerfile .`).
COPY tools/gslot /opt/tools/gslot
COPY tools/layer-distribution /opt/tools/layer-distribution
# COPY tools/fleet-console /opt/tools/fleet-console   # <- when that target lands

# Install both packages into site-packages (no-deps: neither has deps; no
# network fetch: both directories are in the context). This is what makes
# `python3 -m gslot` / `python3 -m layer_distribution` resolve from anywhere.
RUN python3 -m pip install --no-deps --no-cache-dir /opt/tools/gslot && \
    python3 -m pip install --no-deps --no-cache-dir /opt/tools/layer-distribution

# gslot's daemon binds a unix socket (and an optional loopback observation
# port); run it with -v /run so the socket is shared with whoever needs slots.
ENTRYPOINT ["python3", "-m", "gslot"]
CMD ["--help"]
