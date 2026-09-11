# Hugging Face Model Cards — Architecture Support Status

This document is the authoritative "what works, where, and with what caveats"
reference for the architectures added in this fork. Use it as the source for
the **Support** / **Compatibility** section of model cards published to
Hugging Face (and other model hubs).

Status reflects the publication snapshot at `fork-base`. Each entry states:

- **head/tail split** — whether multi-stage (pipeline) execution has been
  verified for the architecture.
- **backends** — which compute backends are verified, and with which caveats.

Cross-references to `meta#NN` are internal tracking handles; every entry below
is self-contained, so a reader does not need to resolve the handle to act on it.
See [KNOWN-ISSUES.md](KNOWN-ISSUES.md) for the known-bad cases.

## Summary

| Architecture | head/tail split | Backends |
|---|---|---|
| `deepseek4` | verified (pending numeric check) | CUDA verified; Vulkan — AMD RDNA4 (RADV, CPU-exact), NVIDIA (coopmat1, CPU-exact), Intel ANV (self-consistent); AMD RDNA3 known-bad in this snapshot (Vulkan flash attention, see `meta#85`) |
| `qwen4exp` | verified | CUDA verified; Vulkan — AMD RDNA4 (RADV, CPU-exact), NVIDIA (coopmat1, CPU-exact), Intel ANV (self-consistent); AMD RDNA3 known-bad in this snapshot (Vulkan flash attention, see `meta#85`) |
| `glm5next` | verified | CUDA verified; Vulkan — AMD RDNA4 (RADV, CPU-exact), NVIDIA (coopmat1, CPU-exact), Intel ANV (self-consistent); AMD RDNA3 known-bad in this snapshot (Vulkan flash attention, see `meta#85`) |
| `gemma4` | self-consistent (both stages agree) | CPU verified (Q4_K embedding); CUDA batched prefill known-bad (`meta#81`); Q6_K embedding known-bad (`meta#80`); Vulkan — AMD RDNA4 (RADV, CPU-exact), NVIDIA (coopmat1, CPU-exact), Intel ANV (self-consistent); AMD RDNA3 known-bad in this snapshot (Vulkan flash attention, see `meta#85`) |

### Card wording (drop-in)

> **head/tail split verified:** deepseek4 (pending numeric check), qwen4exp,
> glm5next; **backends:** CUDA verified, Vulkan verified — AMD RDNA4 (RADV, CPU-exact),
> NVIDIA (coopmat1, CPU-exact), Intel ANV (self-consistent); AMD RDNA3 known-bad in this
> snapshot (Vulkan flash attention, see `meta#85`).

Use the line above verbatim for the three archs it names. For `gemma4`, use the
per-arch note below instead, because its support is conditional on the quant
variant and the backend.

## Per-architecture notes

### `deepseek4`
- **head/tail split:** verified. A final numeric (byte-identical token) check
  against the single-process reference is still pending; the split machinery
  itself is confirmed.
- **backends:** CUDA verified. Vulkan verified — AMD RDNA4 (RADV, CPU-exact),
  NVIDIA (coopmat1, CPU-exact), Intel ANV (self-consistent); AMD RDNA3 known-bad in this
  snapshot (Vulkan flash attention, see `meta#85`).

### `qwen4exp`
- **head/tail split:** verified.
- **backends:** CUDA verified. Vulkan verified — AMD RDNA4 (RADV, CPU-exact),
  NVIDIA (coopmat1, CPU-exact), Intel ANV (self-consistent); AMD RDNA3 known-bad in this
  snapshot (Vulkan flash attention, see `meta#85`).

### `glm5next`
- **head/tail split:** verified.
- **backends:** CUDA verified. Vulkan verified — AMD RDNA4 (RADV, CPU-exact),
  NVIDIA (coopmat1, CPU-exact), Intel ANV (self-consistent); AMD RDNA3 known-bad in this
  snapshot (Vulkan flash attention, see `meta#85`).

### `gemma4`
- **head/tail split:** the two-stage loopback is self-consistent (head and tail
  agree with each other), so the split machinery is sound for this arch. The
  output correctness below is dominated by backend/quant issues, not by the
  split.
- **backends:**
  - **CPU:** verified, **only with a Q4_K `token_embd`** (e.g. the Unsloth
    `gemma-4-12b-it-Q4_0` quant). A `Q6_K` `token_embd` (e.g. Google's official
    QAT `gemma-4-12b-it-qat-q4_0.gguf`) currently produces degenerate output —
    see `meta#80`.
  - **CUDA:** batched prefill is known-bad — the default `-fa on` batched path
    produces wrong tokens, while `-ngl 0` (CPU) is correct. See `meta#81`.
  - **Vulkan:** verified — AMD RDNA4 (RADV, CPU-exact), NVIDIA (coopmat1, CPU-exact),
    Intel ANV (self-consistent); AMD RDNA3 known-bad in this snapshot (Vulkan flash
    attention, see `meta#85`).
- **Card guidance:** until `meta#80` and `meta#81` are resolved, publish the
  `gemma4` card with the CPU + Q4_K-embedding path as the verified configuration
  and state the CUDA/Vulkan caveats explicitly.

## How to read "verified"

- **verified** — the stated path produced the expected (reference-matched)
  output on the stated backend in this snapshot.
- **pending numeric check** — the mechanism is confirmed and the output is
  coherent, but a byte-identical token comparison against the single-process
  reference has not yet been captured.
- **self-consistent** — the multi-stage run agrees with itself (stages match),
  which validates the transport/split but does not by itself prove the tokens
  are correct; correctness is stated separately per backend.
- **known-bad** — the path is reproduced-wrong in this snapshot and is tracked
  in [KNOWN-ISSUES.md](KNOWN-ISSUES.md).
