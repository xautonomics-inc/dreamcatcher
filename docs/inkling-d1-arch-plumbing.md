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
  5:1 pattern and `n_swa = 512`; set `recurrent_layer_arr` (Inkling is
  all-recurrent for short-conv); compute `n_embd_r_impl` if needed.
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

## Open questions / notes for review

- Whether the rel-extent window needs a new `llama_kv_cache` field or can be
  derived from the existing SWA window. Flag with file:line if a gap appears.
- Whether short-conv state needs a distinct `s_l` layout from qwen3next's, or can
  reuse it as-is. Flag with file:line if a gap appears.