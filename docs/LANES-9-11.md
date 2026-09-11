# Lane 9–11 Documentation

This document describes the three lanes that build on top of lane 8 (the upstream graft lane)
in the public replay sequence.

## Lane 9: Vulkan backend refresh

**Source branch:** `the Vulkan backend refresh branch`
**Status:** Merged into `fork-base` (commit `ea5bd7e7`)
**Date:** 2026-09-11

### Summary

The internal Vulkan backend was grafted from the 2026 mainline ggml and adapted to ik. This
replaces the pre-graft backend (last synced from mainline in mid-2025) which had drifted far
enough to produce wrong output on several devices rather than a clean refusal.

### What changed

- **`ggml/src/ggml-vulkan.cpp`** and **`ggml/src/vulkan-shaders/`**: Replaced with 2026 mainline
  backend code, adapted to ik via shims in `ggml-vulkan-ik-compat.h`.
- **`ggml/src/ggml-vulkan-ik-compat.h`** (new): Shims for ik's pre-module-split architecture
  (mainline's `ggml-cpu.h`, `GGML_LOG_*`, backend-registry entry points).
- **ik fused ops**: `FUSED_UP_GATE`, `MOE_FUSED_UP_GATE`, `MUL_MULTI_ADD`, `FUSED_MUL_UNARY`
  mapped onto mainline's GLU split path for `SIGMOID`.
- **ik quantization types**: `IQ4_K`, `IQ5_K`, `IQ6_K` with dequant, `get_rows`, `mul_mat_vec`,
  and `mul_mm` kernels.
- **Build dependency**: `spirv-headers` (`spirv-headers` on Debian/Ubuntu).
- **New CMake shader generator**: One translation unit per shader with depfile tracking.

### Device compatibility

| Device | Driver | Status |
|--------|--------|--------|
| AMD RDNA4 | RADV | Verified, CPU-exact |
| NVIDIA | CoopMat1 | Verified, CPU-exact |
| NVIDIA | CoopMat2 | Declined by default; opt-in via `GGML_VK_ENABLE_COOPMAT2=1` |
| Intel | ANV | Self-consistent, not CPU-exact |
| AMD RDNA3 | RADV | Pending verification |

### `-ngl 0` semantics

An explicit `-ngl 0` now disables the auto-fit planner and keeps all layers on the CPU. No
silent GPU offload occurs. This is enforced by restoring ik's `ne[1]` offload predicate.

### Fused graph defaults

When any layer is offloaded to Vulkan:

- The fused up/gate graph defaults **off** (the split graph is used instead) because the fused
  path is 11–17% slower at decode.
- `GGML_VK_ALLOW_FUG=1` opts back into fused up/gate.
- `GGML_VK_ALLOW_FMOE=1` opts back into fused MoE.
- Both choices are logged once at context creation.

### Test results

`test-backend-ops -b Vulkan0` runs to completion: 1384/1648 pass. The remaining failures are
inherited from stock (FA/CPY) and fewer than upstream, which crashes mid-sweep.

### Documentation

See [`docs/VULKAN-BACKEND.md`](VULKAN-BACKEND.md) for the full reference.

## Lane 10: Expert-server port

**Source branch:** `the expert-server port branch`
**Status:** Merged into `fork-base` (commit `ea5bd7e7`)
**Date:** 2026-09-11 (review PASS)

### Summary

Ports the expert-server role and the exactness harness for distributed MoE inference. This
allows routed-expert tensors to be served by a remote expert server rather than being
materialized locally.

### What changed

- **Expert-server role**: New role that serves routed-expert tensors over the network.
- **Exactness harness**: Verification that remote expert computation matches local computation.
- **Attention-side transport client**: Client code for the attention-side transport.
- **Covered MoE layers**: Routed-expert tensors served by a remote server are skipped during
  local computation.
- **`llama-expert-server`**: New binary for running an expert server process.
- **`llama-expert-check`**: New binary for the byte-exactness / overhead harness.
- **Client configuration**: Env-only via `LLAMA_EXPERTS_REMOTE=host:port@layers`; no command-line
  flag exists for this configuration.

### Status

- CPU byte-exactness: verified all-remote on two architectures (x86_64, aarch64).
- GPU proof: pending Toshi's GPU window verification.
- Acceptance plan: see `~/workspace/lane10/ACCEPTANCE-PLAN.md` (P1 digest / P2 expert-check /
  P3 restore).

## Lane 11: gslot runtime

**Status:** Merged into `fork-base` (commit `1c26dd49`)
**Date:** 2026-09-11 (emma's review PASS)

### Summary

The gslot runtime provides a configuration-driven approach to GPU slot allocation and
multi-stage pipeline management. It gates the head/relay/tail/server stages behind a
global slot router: when the arbiter is running, each stage requests a grant before
proceeding; when it is not running, the gate fails open and stages proceed unconditionally.

### What changed

- **`tools/gslot/arbiter.py`** (new): Slot allocation arbiter (daemon). Maintains a pool
  of GPU slots and grants leases to requesting stages via an NDJSON protocol.
- **`tools/gslot-run`** (new): Tenant launcher that registers a lease with the arbiter
  and execs a command pinned to the granted resources.
- **`c/gslot_client.h`** (new): C tenant client (speaks the NDJSON protocol).
- **Gate integration**: `gslot_client.h` is included in the stage-runner; the gate is
  controlled by `STAGE_GSLOT_*` environment variables and defaults **OFF** (fail-open).

### Configuration

The slot router is controlled by environment variables (no command-line flags):

| Variable | Default | Description |
|----------|---------|-------------|
| `STAGE_GSLOT_ENABLE` | `0` | Master switch. When `0` (or unset), the gate is bypassed and stages run unconditionally (fail-open). |
| `STAGE_GSLOT_HOST` | `127.0.0.1` | Arbiter host. |
| `STAGE_GSLOT_PORT` | `9876` | Arbiter port. |
| `STAGE_GSLOT_TIMEOUT` | `5.0` | Timeout (seconds) for lease acquisition. |

### Proof

emma's acceptance proof: two-stage Gemma-4 loopback under a live arbiter, 28 grants /
0 overruns, byte-identical to the reference. The arbiter was started with 4 GPU slots;
each stage requested and held its lease for the duration of the forward pass, then
released it. No stage was starved or overrun.

### Replay sequence

```
lane 9 (Vulkan) → lane 10 (expert-server) → lane 11 (gslot)
```

Lane 9 is merged into `fork-base`. Lanes 10 and 11 are merged into `fork-base` and will
be replayed in sequence.