# D1 - Inkling architecture plumbing (design note, rev 2)

Owner: noah. Branch: `agent/noah/inkling-d1-arch-plumbing`. P0.

Rev 2 corrects the premise after Ben's review (`b0f42e90`): dreamcatcher is NOT
missing SWA or recurrent state. It has ik-native versions of both, so D1 must
NOT port upstream's memory hierarchy. This revision narrows D1 to the arch
plumbing only.

## Corrected premise

Dreamcatcher already has, natively:

- **SWA** — `llama_kv_cache` has `size_swa`/`sink_rows`/`window_swa`/`head_swa`/
  `pos_base_swa` (`src/llama-context.h:106-112`) plus `llama_swa_calc_window_view`
  (`:16-40`). The interleaved pattern is expressed per-layer via
  `hparams.swa_layers[il]` and `hparams.n_swa_pattern`
  (`src/llama-hparams.h:40-41,214`), and Gemma3 uses it through
  `build_inp_KQ_mask_swa` + `llm_build_kv(..., n_swa)` (`build_gemma3.cpp:25,72-74`).
- **Recurrent state** — `llama_kv_cache::recurrent` and `s_l` (per-layer
  recurrent/conv state storage) (`src/llama-context.h:79-85,126`), already used
  by qwen3next ("qnext") and openPangu.
- **Stage emit** — a head stage emitting the post-window residual instead of
  logits is already implemented for hybrid models
  (`build_qwen3next.cpp:18-24,90-93`, via `llm_stage.h`).

Inkling's 5:1 SWA pattern (55 SWA + 11 global, window 512) maps onto
`swa_layers[il] = (il_abs(il) % 6 != 5)` with `n_swa = 512`; its short-conv state
maps onto `recurrent` + `s_l` the way qwen3next does it.

## Decisions (from Ben)

1. **No upstream memory classes.** Do NOT port `llama-kv-cache-iswa`,
   `llama-memory-hybrid-iswa`, `llama-memory-recurrent`, or
   `build_attn_inp_kv_iswa`. Inkling goes through ik's single `llama_kv_cache`:
   the SWA fields for its 5:1 pattern, and `recurrent`/`s_l` for short-conv
   state. If a gap shows up (e.g. both SWA and recurrent at once, or the
   rel-extent window), extend that struct and cite the gap with file:line.
2. **No banded-FA op in D1 or D2.** Correctness in D2 uses the existing SWA mask
   path on CPU. `GGML_OP_FLASH_ATTN_EXT_BANDED` and its CUDA kernel are
   performance work → D3 (gwen).
3. **D1 scope** is exactly the arch plumbing below. Gate: Inkling-Small metadata
   and every tensor created, on CPU, no compute.

## D1 scope (exact)

- `src/llama-arch.h` — add `LLM_ARCH_INKLING` to the arch enum; add Inkling
  tensor names to `enum llm_tensor` (shortconv_k/v/attn/mlp, attn_rel_proj,
  attn_rel_b, attn_rel_b_swa); add `LLM_KV_INKLING_*` metadata keys.
- `src/llama-arch.cpp` — add the `"inkling"` arch name and the Inkling KV key
  string table.
- `src/llama-hparams.h` — add Inkling hparams: `n_shortconv_l_cache` (short-conv
  kernel), `inkling_d_rel`, `inkling_rel_extent`, `inkling_rel_extent_swa`,
  `inkling_log_n_floor`, `inkling_log_alpha`, `inkling_unpadded_n_vocab`, and the
  SWA pattern/window fields (reuse `n_swa`/`n_swa_pattern`/`swa_layers` where
  possible). Add `is_swa()` / `is_recr()` accessors reconciled with the existing
  `swa_layers` / `recurrent_layer_arr`.
- `src/llama-hparams.cpp` — parse the Inkling keys; set `swa_layers[il]` for the
  5:1 pattern and `n_swa = 512`; compute `n_embd_r_impl`. Note: Inkling's
  shortconv state is NOT flagged into `recurrent_layer_arr` — that would drive
  the KV-cache build down the `qnext_recurrent` zero-width path (see the D3
  hand-off below).
- `src/llama-model.h` — add `llama_layer` fields for shortconv_k/v/attn/mlp,
  attn_rel_proj, attn_rel_b, attn_rel_b_swa; declare `llama_model_inkling`.
- `src/llama-model.cpp` — Inkling tensor loader (`load_arch_tensors`), tensor
  name mapping, arch dispatch in the model constructor.
- `src/llama-load-tensors.cpp` — Inkling tensor-name/loader table entries.
- `src/llama-vocab.cpp` / `.h` — Inkling vocab handling (unpadded vocab masking)
  if not already covered by existing paths.
- `src/llama-model-saver.cpp` — Inkling model-saver entries.
- `tests/test-llama-archs.cpp` — add Inkling to the arch round-trip test.

## What D1 deliberately does NOT do

- No `build_inkling()` graph implementation (D2).
- No banded flash-attn op (D3, gwen).
- No upstream memory classes (`llama-kv-cache-iswa`, `llama-memory-hybrid-iswa`,
  `llama-memory-recurrent`, `build_attn_inp_kv_iswa`).
- No stage rings / stage-runner wiring (D4b), no remote experts (D4c).
- No mmproj / multimodal Inkling support (D3/D4a scope; the oracle's
  `tools/mtmd/models/inkling.cpp` and `conversion/inkling.py` are out of D1).
- No HF model card / library publish (D5b).

## Gate for D1

D1 is complete when Inkling-Small's metadata loads and **every tensor is
created** on CPU, with no compute. I will verify with a load-only smoke against
the D0 oracle's Inkling-Small GGUF once it is available, plus the arch
round-trip test. Runtime parity is D2's gate.

## D1 verification (2026-09-16, rev 2)

### Tensor gate: PASSES on the real checkpoint

The D1 gate — Inkling-Small's metadata loads and every tensor is created, on
CPU, no compute — is met. Verified on the real Inkling-Small GGUF (nvidia,
both endpoints stopped): **960 tensors** created with correct shapes, four CPU
buffers, **151.363 GiB**, no `check_tensor_dims` failures, no forward pass.
cody independently confirmed the gate holds: reaching `llama_init_from_model`
requires `done_getting_tensors()`, which throws unless `n_created == n_tensors`
— and that equality held, so every declared tensor was created.

The synthetic model (`tools/make-inkling-test-gguf.py`, 2 layers, 45 tensors)
also loads cleanly.

### KV-cache failure: a real bug, NOT environmental (rev 2 correction)

Rev 1 attributed the KV-cache allocation failure to the CPU backend's 0-byte
device-memory report. That attribution is **wrong** and is corrected here.

The failure is a zero-sized recurrent-state path, not an OS/backend refusal.
Inkling's `llama-hparams.cpp` flagged every layer recurrent
(`recurrent_layer_arr[il] = true`), so the KV-cache build took the
`qnext_recurrent` branch (src/llama.cpp:1401-1413) and sized `cache_s_l` with
width `n_embd_v_s() + n_embd_ple_conv(i)` = **0** — both derive from
`ssm_d_conv`/`ssm_d_inner`/`ssm_n_group`, which Inkling never sets.
`ggml_nbytes` returns 0 for a nonpositive dimension, `ggml-alloc` allocates no
buffer and returns NULL, and src/llama.cpp:1656 reports that as an allocation
failure.

Single-variable control on the synthetic model (no endpoint time):
- `recurrent_layer_arr[il] = true` → `llama_kv_cache_init: failed to allocate
  buffer for kv cache` (rc=1).
- Same model/binary, flag false → `CPU KV buffer size = 1.00 MiB`,
  `KV self size = 1.00 MiB` allocates fine, then fails at
  `llama-build-context.cpp:3123` (missing `build_inkling`) with rc=134 — i.e. it
  runs to exactly D1's own boundary, the missing graph.

A 14 MB model needing 12 MiB cannot fail for memory reasons; and gemma-4-12b
allocates a 168 MiB cache on the same binary with identical "0 MiB free"
warnings, so those warnings are benign.

**Fix applied:** the `recurrent_layer_arr[il] = true` flag is removed from the
Inkling case (src/llama-hparams.cpp). D1's scope is arch plumbing; the hybrid
attention+recurrent cache is D3's lane. With the flag gone, D1 runs to exactly
its boundary — the missing graph — where it is supposed to stop.

### `n_embd_r_impl`

Kept — it is the right value and D3 will need it. Note that **no consumer
exists yet**: `n_embd_r()` (src/llama-hparams.h:404) has exactly one occurrence
in the tree, its own definition; it is not read by the KV-cache build or the
graph. D3 (gwen) must wire it up when it implements the recurrent-state path.

Two notes on the D1 scope list: `src/llama-model-saver.cpp` and
`tests/test-llama-archs.cpp` do not exist in the dreamcatcher tree (they are
upstream-only). The saver path is not needed for load-only D1 and the arch
round-trip test is not present to extend; both remain D5b/upstream-sync scope.

## Hand-off to D3 (gwen): the recurrence question

The hybrid attention+recurrent cache is D3's lane. Stated problem to inherit,
with file:line evidence:

- Inkling's packed shortconv state is currently **not** wired into the
  layer-library recurrent path. The `recurrent_layer_arr[il] = true` flag was
  removed from src/llama-hparams.cpp because flagging all layers recurrent makes
  the KV-cache build take the `qnext_recurrent` branch (src/llama.cpp:1401-1413),
  which sizes `cache_s_l` from `n_embd_v_s()`/`n_embd_ple_conv()` — both derive
  from `ssm_d_conv`/`ssm_d_inner`/`ssm_n_group`, which Inkling never sets, so the
  width is 0 and `ggml-alloc` returns NULL (src/llama.cpp:1656). D3 must either
  set the ssm geometry Inkling needs or give Inkling its own recurrent-state
  sizing path.
- `n_embd_r_impl` (src/llama-hparams.h:64, computed at
  src/llama-hparams.cpp:781) is the correct shortconv-state width but has no
  consumer yet; D3 must read it from the KV-cache build / graph.
- Whether short-conv state needs a distinct `s_l` layout from qwen3next's, or can
  reuse it as-is, is open. Flag with file:line if a gap appears.
- Whether the rel-extent window needs a new `llama_kv_cache` field or can be
  derived from the existing SWA window is open. Flag with file:line if a gap
  appears.