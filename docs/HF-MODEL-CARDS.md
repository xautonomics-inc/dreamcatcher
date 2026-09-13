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

> [!NOTE]
> Only model files downloaded from [huggingface.co/xautonomics](https://huggingface.co/xautonomics) are supported. Other GGUFs, including libraries you slice yourself, may load but are unsupported.

## Summary

| Architecture | head/tail split | Backends |
|---|---|---|
| `deepseek4` | verified | CUDA monolith **verified** on UD-Q4_K_XL (revision `37044a3c`, tensor-identical to the measured copy; 1328/1328 tensor hashes; full 43-layer forward pass byte-identical, md5-equal hidden states). Library windows fixed (`meta#92`): ring token-identical 64/64 on CUDA with the meta#92 fix; `llama-server --model-dir` is available (`meta#90`; not yet measured on this architecture). Vulkan: degenerate on NVIDIA Vulkan (coopmat1); RDNA3/ANV not measured; not quant-related (`meta#96`) |
| `qwen4exp` | verified (pre-publish); CUDA ring not token-exact on `20c308ca` (see backends) | CUDA **verified** on `20c308ca`: monolith coherent 23.2 tok/s (`llama-server`), library ring coherent 17.4 tok/s, full-model forward pass byte-identical mono-vs-library (1224/1224 hashes); the ring is **not token-exact** — the tail drops `per_layer_token_embd` for a non-zero window (`meta#91`). Vulkan: **fixed** (was degenerate on every vendor) — root cause was a short `MULTI_ADD` `src0` descriptor range; NVIDIA coopmat1 verified coherent, agreeing with CUDA within rounding; **RDNA3 re-measure pending** (`meta#88`) |
| `glm5next` | verified | CUDA **verified** on `20c308ca`: library head+tail ring generates coherently, 7.6 tok/s greedy (UD-IQ4_XS, 1383/1383 hashes vs monolith); the mono-vs-library runtime A/B is **done** — hidden states md5-identical mono vs library (the `GGML_ASSERT(lctx.logits != nullptr)` abort under `STAGE_EMIT=hidden` is fixed, `meta#93`). `--role server` times out awaiting a return edge (`meta#90`). Vulkan (any vendor): not measured for this arch |
| `gemma4` | self-consistent (both stages agree) | CPU verified (Q4_K embedding); CUDA batched prefill known-bad (`meta#81`); Q6_K embedding known-bad (`meta#80`); Vulkan: measured on this model at the lane-9 checkpoint (pre-`20c308ca`) — RDNA4 and RDNA3 (RADV) FA sweep clean and token-exact on chat-formatted prompts, bare greedy prompt can differ from CPU by rounding (`meta#85`); NVIDIA (coopmat1): CPU-exact; Intel ANV: self-consistent, not CPU-exact; AMD RDNA3.5 (8060S APU): verified for the expert-server path with GLM-5.3-Flash experts |

### Card wording (drop-in, per architecture)

The shared line this section used to publish ("backends: CUDA verified, Vulkan
verified — AMD RDNA4 and RDNA3 …") restated one arch's sweep results as all
archs' and has been retired; see `meta#88` for how it got falsified. Use the
per-arch lines below — measured on `20c308ca` plus the relevant merged fix branches (meta#90/92). Hardware:
CUDA: RTX 5060 Ti sm_120; Vulkan RDNA3: RX 7900 XT x4 RADV; Vulkan NVIDIA:
RTX 5060 Ti x4 coopmat1.

> **deepseek4 — CUDA verified.** Monolith verified on CUDA (UD-Q4_K_XL, revision
> `37044a3c`, tensor-identical to the measured copy; 1328/1328 tensor hashes, full
> 43-layer forward pass byte-identical). Library windows fixed (`meta#92`): ring
> token-identical 64/64 on CUDA with the meta#92 fix; `llama-server --model-dir`
> is available (`meta#90`; not yet measured on this architecture). **Vulkan:**
> degenerate on NVIDIA Vulkan (coopmat1); RDNA3/ANV not measured; not quant-related
> (`meta#96`).

> **qwen4exp — CUDA verified** (monolith coherent 23.2 tok/s; library head/tail
> ring coherent 17.4 tok/s, **not token-exact** — the tail drops
> `per_layer_token_embd` for a non-zero window, `meta#91`; library ≡ monolith
> tensor-wise and forward-pass byte-identical). **Vulkan fixed** (was degenerate
> on every vendor — AMD RDNA3/RADV and NVIDIA coopmat1 alike; the cause was a
> short `MULTI_ADD` `src0` descriptor range, not flash attention or BF16 tensor
> placement): NVIDIA coopmat1 verified coherent and agreeing with CUDA within
> rounding; RDNA3 re-measure pending; RDNA4 / Intel ANV not measured for this
> architecture (`meta#88`).

> **glm5next — CUDA verified** (library head/tail ring coherent, greedy,
> 7.6 tok/s, experts on CPU; library ≡ monolith 1383/1383 hashes). The
> mono-vs-library runtime A/B is done — hidden states md5-identical mono vs
> library (the `STAGE_EMIT=hidden` logits abort that blocked it is fixed,
> `meta#93`). Vulkan: not measured for this architecture on the published tree
> (the RDNA3.5 APU proof was the expert-server path with GLM-5.3-Flash experts,
> not a local Vulkan run).

Common to all three: a layer library is loaded by `llama-server --model-dir`
for single-process serving, or partitioned across `llama-stage-runner`
(`--model-dir`) processes for a distributed ring. On Vulkan hosts with
big pinned tensors, use `LLAMA_NO_HOST_OVERRIDES=1` (see `meta#89` for the
recipe corrections).

For `gemma4`, use the per-arch note below instead, because its support is
conditional on the quant variant and the backend.

## Per-architecture notes

### `deepseek4`
- **head/tail split:** verified — library windows fixed (`meta#92`: loader window
  validation and compression ratios sliced for stage window). Ring token-identical
  64/64 on CUDA with the meta#92 fix.
- **backends:** CUDA monolith **verified** (production serving, and UD-Q4_K_XL
  revision `37044a3c` [tensor-identical to the measured copy] library ≡ monolith:
  1328/1328 tensor hashes and byte-identical full 43-layer forward pass with
  md5-equal hidden states). `llama-server --model-dir` is available (`meta#90`);
  not yet measured on this architecture. Vulkan: degenerate on NVIDIA Vulkan
  (coopmat1); RDNA3/ANV not measured; not quant-related (`meta#96`).
- **revision pin (`meta#102`):** Current upstream Unsloth revisions (≥ `e1efe867`,
  2026-09-04) declare 43 extra `exp_probs_b_vl.bias` vision tensors that the fork
  loader does not construct, which from code inspection causes the loader to abort with
  "wrong number of tensors" (expected 1371, got 1328; unrun on live hardware). The
  supported distribution is the published library built from pinned revision
  `37044a3c`.

### `qwen4exp`
- **head/tail split:** verified (mechanism); the CUDA ring run on `20c308ca`
  is coherent but not token-exact vs the monolith because the tail drops
  `per_layer_token_embd` for a non-zero window (`meta#91`).
- **backends:** CUDA **verified** on `20c308ca`: monolith via `llama-server`
  coherent at 23.2 tok/s; library head+tail ring coherent at 17.4 tok/s;
  library ≡ monolith — 1224/1224 tensor hashes and a byte-identical full
  forward pass. Vulkan: **fixed** (was degenerate on all vendors — UD-IQ4_XS
  emitted `**:** **:**;**;**…` on AMD RDNA3 (RX 7900 XT x4, RADV; full offload
  and `-ngl 12`, flash attention on and off) and identically on NVIDIA
  (4x 5060 Ti, coopmat1)). The cause was a short `MULTI_ADD` `src0` descriptor
  range in the generic dispatcher — not flash attention, not BF16 tensor
  placement (both refuted during triage). NVIDIA coopmat1 verified coherent
  after the fix, agreeing with CUDA within rounding; **RDNA3 re-measure
  pending** (the fix is device independent; run `test-backend-ops -o MULTI_ADD`
  there). CPU remains coherent (`meta#88`).
  Vulkan on AMD RDNA4 / Intel ANV: not measured for this architecture on the
  published tree (the RDNA4 sweeps on record were Gemma-4 12B and Qwen3 8B).

### `glm5next`
- **head/tail split:** verified — on `20c308ca` the library head+tail ring
  generates coherently (greedy, 7.6 tok/s, experts on CPU across 4 GPUs).
- **backends:** CUDA **verified** as above; library ≡ monolith 1383/1383
  tensor hashes, and the runtime mono-vs-library A/B is **done**: with the
  `STAGE_EMIT=hidden` logits abort fixed (`GGML_ASSERT(lctx.logits != nullptr)`
  when `nextn_predict_layers>0`), hidden states are md5-identical mono vs
  library (`meta#93`). `--role server` with `ci/smoke-serve.sh` times out
  awaiting a return edge nothing sends (`meta#90`). Vulkan (any vendor): not
  measured for this architecture on the published tree (the RDNA3.5 APU proof
  was the expert-server path serving GLM-5.3-Flash experts to a CUDA head, not
  a local Vulkan run; the RDNA3/RDNA4 sweeps on record were Gemma-4 12B and
  Qwen3 8B).

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
