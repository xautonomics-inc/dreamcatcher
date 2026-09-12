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
| `deepseek4` | verified | CUDA monolith **verified** on `20c308ca` (UD-Q4_K_XL; library-vs-monolith full-model CUDA forward pass byte-identical, md5-equal hidden states; 1328/1328 tensor hashes). Library windows **fixed (`meta#92`)**: head+tail ring vs monolith **token-identical 64/64**; single-process via `llama-server --model-dir` (with `meta#90`). Vulkan NVIDIA (coopmat1): **open, `meta#96`** — degenerate on `20c308ca` while CUDA is coherent; not the quant (MXFP4 accepted natively), not the `meta#88` MULTI_ADD bug (DSv4 never emits that op with experts on CPU); a separate Vulkan dim-0 `GET_ROWS` defect found alongside is fixed (MR !43), DSv4 remains degenerate — third fault, check queued. Vulkan RDNA3/RDNA4: not measured for this arch |
| `qwen4exp` | verified (pre-publish); CUDA ring not token-exact on `20c308ca` (see backends) | CUDA **verified** on `20c308ca`: monolith coherent 23.2 tok/s (`llama-server`), library ring coherent 17.4 tok/s, full-model forward pass byte-identical mono-vs-library (1224/1224 hashes); the ring is **not token-exact** — the tail drops `per_layer_token_embd` for a non-zero window (`meta#91`). Vulkan: **fixed (MULTI_ADD full-chunk-range bind, `meta#88`)** — NVIDIA coopmat1 verified coherent, agrees with CUDA within rounding; RDNA3 re-measure pending |
| `glm5next` | verified | CUDA **verified** on `20c308ca`: library head+tail ring generates coherently, 7.5 tok/s greedy (UD-IQ4_XS, 1383/1383 hashes vs monolith). The hidden-emit A/B abort (`GGML_ASSERT(lctx.logits != nullptr)`, `meta#93`) is **fixed on `fork-base`** (MR !40, unified logits predicate + truth-table test) — with it, GLM-5.3 library-vs-monolith hidden state is **byte-identical (md5-equal)** and generation is unchanged. `--role server` still times out awaiting a return edge (`meta#90`). Vulkan NVIDIA (coopmat1): **measured coherent** on `20c308ca` (UD-IQ4_XS, 1.9 tok/s tg with experts on CPU). Vulkan RDNA3/RDNA4: not measured for this arch |
| `gemma4` | self-consistent (both stages agree) | CPU verified (Q4_K embedding); CUDA batched prefill known-bad (`meta#81`); Q6_K embedding known-bad (`meta#80`); Vulkan: measured on this model at the lane-9 checkpoint (pre-`20c308ca`) — RDNA4 and RDNA3 (RADV) FA sweep clean and token-exact on chat-formatted prompts, bare greedy prompt can differ from CPU by rounding (`meta#85`); NVIDIA (coopmat1): CPU-exact; Intel ANV: self-consistent, not CPU-exact; AMD RDNA3.5 (8060S APU): verified for the expert-server path with GLM-5.3-Flash experts |

### Card wording (drop-in, per architecture)

The shared line this section used to publish ("backends: CUDA verified, Vulkan
verified — AMD RDNA4 and RDNA3 …") restated one arch's sweep results as all
archs' and has been retired; see `meta#88` for how it got falsified. Use the
per-arch lines below — all measured on the published tree `20c308ca`. Hardware:
CUDA: RTX 5060 Ti sm_120; Vulkan RDNA3: RX 7900 XT x4 RADV; Vulkan NVIDIA:
RTX 5060 Ti x4 coopmat1.

> **deepseek4 — quant verified, library windows fixed.** CUDA monolith verified
> (production serving); the UD-Q4_K_XL library is exact vs its monolith
> (1328/1328 tensor hashes, full 43-layer forward pass byte-identical), and
> library windows are **fixed (`meta#92`)**: head+tail ring vs monolith is
> **token-identical 64/64** on CUDA, and single-process serving works via
> `llama-server --model-dir` (with `meta#90`). Vulkan NVIDIA (coopmat1):
> **open, `meta#96`** — degenerate on `20c308ca` while CUDA is coherent; not
> the quant (MXFP4 accepted natively), not the `meta#88` MULTI_ADD bug (DSv4
> never emits that op with experts on CPU); a separate Vulkan `GET_ROWS` defect
> found alongside it is fixed, DSv4 remains degenerate — third fault, check
> queued. Other Vulkan vendors: not measured for this architecture.

> **qwen4exp — CUDA verified** (monolith coherent 23.2 tok/s; library head/tail
> ring coherent 17.4 tok/s, **not token-exact** — the tail drops
> `per_layer_token_embd` for a non-zero window, `meta#91`; library ≡ monolith
> tensor-wise and forward-pass byte-identical). **Vulkan: fixed (MULTI_ADD
> full-chunk-range bind, `meta#88`)** — NVIDIA coopmat1 verified
> coherent and agreeing with CUDA within rounding order; RDNA3 re-measure
> pending. RDNA4 / Intel ANV: not measured for this architecture.

> **glm5next — CUDA verified** (library head/tail ring coherent, greedy,
> 7.5 tok/s, experts on CPU; library ≡ monolith 1383/1383 hashes; the
> single-process A/B is byte-identical after the hidden-emit fix, `meta#93`).
> No single-process server for a library (`meta#90`).
> Vulkan NVIDIA (coopmat1): **measured coherent** on `20c308ca` (1.9 tok/s tg,
> experts on CPU). Other Vulkan vendors: not measured for this architecture
> (the RDNA3.5 APU proof was the expert-server path with GLM-5.3-Flash
> experts, not a local Vulkan run).

Common to all three: a layer library is loaded by `llama-stage-runner`
(`--model-dir`) and, since `meta#90` was fixed, also in-process by
`llama-server --model-dir`; a head+tail ring is still needed for split
serving, and on Vulkan hosts with big pinned tensors set
`LLAMA_NO_HOST_OVERRIDES=1` (see `meta#89` for the recipe corrections).
`--model` with a directory path remains a silent failure mode — there is no
error and no fallback; check the path before blaming the split.

For `gemma4`, use the per-arch note below instead, because its support is
conditional on the quant variant and the backend.

## Per-architecture notes

### `deepseek4`
- **head/tail split:** **verified** — on `20c308ca` the library head+tail ring
  is **token-identical to the monolith, 64/64** (CUDA, UD-Q4_K_XL). The former
  window failure was metadata, not the quant: the converter's
  `compress_ratios` array (46 entries for 43 blocks) was never sliced for the
  window and `hash_layer_count` was never rebased, so any window not starting
  at block 0 built the wrong layers; fixed in the loader with a window
  validator that turns the old segfault into a named error (`meta#92`).
- **backends:** CUDA monolith **verified** — production serving, and on
  `20c308ca` the UD-Q4_K_XL library ≡ monolith: 1328/1328 tensor hashes and a
  byte-identical full 43-layer forward pass (md5-equal hidden states).
  Single-process serving of a library works via `llama-server --model-dir`
  (`meta#90`). Vulkan NVIDIA (coopmat1): **open — `meta#96`**: degenerate on
  `20c308ca` while CUDA is coherent; not the quant (MXFP4 accepted natively,
  same buffer sizes), and **not** the `meta#88` MULTI_ADD bug (DSv4 never
  emits that op with experts on CPU). A second Vulkan defect found during the
  hunt — ik's dim-0 `GET_ROWS` silently computed as a plain gather — is fixed
  (MR !43, now declined), but DSv4 stays degenerate: a third fault,
  check-results pass queued. Vulkan RDNA3/RDNA4: not measured for this
  architecture on the published tree.

### `qwen4exp`
- **head/tail split:** verified (mechanism); the CUDA ring run on `20c308ca`
  is coherent but not token-exact vs the monolith because the tail drops
  `per_layer_token_embd` for a non-zero window (`meta#91`).
- **backends:** CUDA **verified** on `20c308ca`: monolith via `llama-server`
  coherent at 23.2 tok/s; library head+tail ring coherent at 17.4 tok/s;
  library ≡ monolith — 1224/1224 tensor hashes and a byte-identical full
  forward pass. Vulkan: **fixed (MULTI_ADD full-chunk-range bind)** — the degenerate UD-IQ4_XS
  output seen on every vendor (AMD RDNA3 RADV and NVIDIA coopmat1, flash
  attention on and off) was the `MULTI_ADD` op binding a view-sized descriptor
  range: the shader read chunks 1..n of the last row out of bounds (zeros), so
  each hyper-connection mix returned one stream instead of the 4-stream mean.
  Fixed by binding the full chunk range (op tests 7/7 fail → pass); NVIDIA
  coopmat1 monolith now coherent and CUDA-agreeing within rounding; RDNA3
  re-measure pending (`meta#88`). Vulkan on AMD RDNA4 / Intel ANV: not
  measured for this architecture on the published tree.

### `glm5next`
- **head/tail split:** verified — on `20c308ca` the library head+tail ring
  generates coherently (greedy, 7.5 tok/s, experts on CPU across 4 GPUs).
- **backends:** CUDA **verified** as above; library ≡ monolith 1383/1383
  tensor hashes. The hidden-emit A/B that was blocked by an abort (any GLM
  load under `STAGE_EMIT=hidden` with `nextn_predict_layers>0` tripping
  `GGML_ASSERT(lctx.logits != nullptr)`) is **fixed** — two
  disagreeing "do we have logits?" predicates (a stale `nextn_predict_layers`
  proxy vs the real `has_mtp` context flag) were unified into one policy with
  a truth-table test (`meta#93`, MR !40); with it, GLM-5.3 library-vs-monolith
  hidden state is **byte-identical (md5-equal)** and generation is unchanged.
  `--role server` on a library was a timeout awaiting a return edge nothing
  sent (`meta#90`); fixed — the server now loads libraries in-process via
  `--model-dir` (verified for deepseek4; a GLM-specific single-process run is
  not yet on record). Vulkan NVIDIA (coopmat1): **measured coherent** on `20c308ca`
  (UD-IQ4_XS, 1.9 tok/s tg with experts on CPU). Other Vulkan vendors: not
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
