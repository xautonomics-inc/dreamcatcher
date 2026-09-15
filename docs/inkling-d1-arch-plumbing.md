# D1 - Inkling architecture plumbing (design note)

Owner: noah. Branch: `agent/noah/inkling-d1-arch-plumbing`. P0.

Goal of D1: add the Inkling architecture plumbing to dreamcatcher so a later lane
(D2) can write `build_inkling.cpp` and validate against the D0 oracle. D1 itself
must land the arch enum, tensor names, hparams, loader/quant hooks, and the two
ggml operators and memory/attention infrastructure Inkling depends on. It does
NOT build the Inkling graph yet - that is D2.

The oracle for everything below is the internal llama.cpp-lineage port on
`agent/noah/inkling-p1` (matches upstream PR #25731 on Inkling-Small, 48/48
tokens identical). The reference commit is `ce16fff2a` ("Add TML Inkling
architecture"), plus the later iswa/banded/experts-remote work.

## What Inkling needs that dreamcatcher does not have yet

Dreamcatcher is a heavily refactored fork. Compared to the oracle, it is missing
the whole iswa/hybrid-memory/recurrent-state subsystem and the banded flash
attention op. Concretely:

- No `src/llama-kv-cache-iswa.*` / `src/llama-memory-hybrid-iswa.*` /
  `src/llama-memory-recurrent.*`. Dreamcatcher keeps its state in the
  `llama_kv_cache` struct in `src/llama-context.h` (fields `k_l`, `v_l`, `s_l`)
  and its graph builders under `src/graphs/`, and there is no
  `llama_memory_i` / `llama_memory_context_ptr` interface.
- No `GGML_OP_FLASH_ATTN_EXT_BANDED` op. `ggml/include/ggml.h` only has
  `GGML_OP_FLASH_ATTN_EXT` and `GGML_OP_FLASH_ATTN_BACK`.
- No Inkling tensor names in `enum llm_tensor` (`src/llama-arch.h`), no
  `LLM_ARCH_INKLING` in the arch enum, no Inkling hparams fields.
- No `build_attn_inp_kv_iswa()` / `llm_graph_input_attn_kv_iswa` helper. The
  gemma3/4 builders here use `build_std_attention` directly rather than the
  oracle's iswa attention input path.

## Files I will touch (with what each does)

### ggml layer - banded flash attention op

- `ggml/include/ggml.h` - add `GGML_OP_FLASH_ATTN_EXT_BANDED` to the op enum
  and declare `ggml_flash_attn_ext_banded(...)`.
- `ggml/src/ggml.c` - add the op case, tensor name ("flash_attn_ext_banded"),
  the `ggml_flash_attn_ext_banded` constructor, and the
  `GGML_OP_FLASH_ATTN_EXT_BANDED` handling in the compute/dup/check paths.
- `ggml/src/ggml-cpu/ops.cpp` - CPU fallback for the banded op (or route to the
  existing FLASH_ATTN_EXT CPU path with the banded mask).
- `ggml/src/ggml-cuda/fattn-banded.cu` / `.cuh` - the banded MMA flash attention
  kernel (ported from the oracle; fp16 accumulator overflow guard).
- `ggml/src/ggml-cuda/fattn-common.cuh`, `fattn-mma-f16.cuh`, `fattn.cu`,
  `ggml-cuda.cu` - wire the banded kernel into the CUDA dispatch.
- `ggml/src/ggml-cuda/mmf.cuh`, `mmq.cuh`, `mmvf.cu`, `mmvq.cu`, `pad.cu`,
  `ssm-conv.cu`, `argsort.cu` - the small CUDA fixes the oracle carries alongside
  the banded op (these are needed for the op to build/run; ported verbatim).
- `ggml/src/ggml-backend-meta.cpp`, `ggml/src/ggml-rpc/ggml-rpc.cpp`,
  `ggml/include/ggml-rpc.h` - op registration for meta/rpc backends.

### arch / hparams / model plumbing

- `src/llama-arch.h` - add `LLM_ARCH_INKLING` to the arch enum, add the Inkling
  tensor names to `enum llm_tensor` (shortconv_k/v/attn/mlp, attn_rel_proj,
  attn_rel_b, attn_rel_b_swa), add `LLM_KV_INKLING_*` metadata keys.
- `src/llama-arch.cpp` - add the `"inkling"` arch name and the Inkling KV key
  string table.
- `src/llama-hparams.h` - add `n_shortconv_l_cache`, `inkling_d_rel`,
  `inkling_rel_extent`, `inkling_rel_extent_swa`, `inkling_log_n_floor`,
  `inkling_log_alpha`, `inkling_unpadded_n_vocab`; add the `is_swa_impl` /
  `is_recr_impl` arrays and `is_swa()` / `is_recr()` accessors (dreamcatcher
  currently has `recurrent_layer_arr` + `swa_layers`; reconcile with these).
- `src/llama-hparams.cpp` - parse the Inkling keys, set `is_recr_impl` /
  `is_swa_impl` uniformly (Inkling is all-recurrent), compute `n_embd_r_impl`.
- `src/llama-model.h` - add `llama_layer` fields: `shortconv_k/v/attn/mlp`,
  `attn_rel_proj`, `attn_rel_b`, `attn_rel_b_swa`; add the `llama_model_inkling`
  class declaration.
- `src/llama-model.cpp` - Inkling tensor loader (`load_arch_tensors`), tensor
  name mapping, and the arch dispatch in the model constructor.
- `src/llama-load-tensors.cpp` - any tensor-name/loader table entries Inkling
  needs.
- `src/llama-quant.cpp` - the Inkling quant hooks the oracle carries.
- `src/llama-vocab.cpp` / `.h` - Inkling vocab handling (unpadded vocab masking)
  if not already covered by existing paths.
- `src/llama-model-saver.cpp` - Inkling model-saver entries.

### attention / memory infrastructure

- `src/llama-kv-cache-iswa.h` / `.cpp` - the two-instance (SWA / non-SWA) KV
  cache for interleaved SWA attention. Ported from the oracle; must adapt to
  dreamcatcher's `llama_kv_cache` (which lives in `src/llama-context.h`).
- `src/llama-memory-hybrid-iswa.h` / `.cpp` - the hybrid (SWA + base) memory
  context and its attention input helper.
- `src/llama-memory-recurrent.h` / `.cpp` - per-layer recurrent shortconv state
  storage (the `s_l` path Inkling uses for its packed shortconv streams).
- `src/llama-build-context.h` / `.cpp` - add the iswa attention input builder
  (`build_attn_inp_kv_iswa` / `llm_graph_input_attn_kv_iswa`) and the banded
  attention call site; add `build_inkling()` declaration. Dreamcatcher's
  `build_std_attention` is the base to extend, mirroring the oracle's gemma3/4
  iswa builders.
- `src/graphs/` - the gemma3/4 iswa builders referenced in the plan are the
  oracle's `gemma3.cpp` / `gemma4.cpp`; in dreamcatcher the equivalent is
  `src/graphs/build_gemma3.cpp` / `build_gemma4.cpp`, which I will extend to
  route through the iswa attention path (the plan's "reuse gemma3/4 iswa
  builders" means reusing the iswa attention pattern they establish, not
  copying the oracle's graph files).

### tests

- `tests/test-llama-archs.cpp` - add Inkling to the arch round-trip test.
- `tests/test-backend-ops.cpp` - add a banded flash-attn op case (the oracle has
  `test-flash-attn-bias.cpp` / `test-flash-attn-generic-hash.cpp`; D6's
  synthetic-GGUF fixture will exercise the full graph).

## What D1 deliberately does NOT do

- No `build_inkling()` graph implementation (D2).
- No stage rings / stage-runner wiring (D4b), no remote experts (D4c).
- No mmproj / multimodal Inkling support (D3/D4a scope; the oracle's
  `tools/mtmd/models/inkling.cpp` and `conversion/inkling.py` are out of D1).
- No HF model card / library publish (D5b).

## Gate for D1

D1 is "arch plumbing" - it is complete when the Inkling arch loads a real
Inkling GGUF's metadata and tensor names without error and the banded op is
present in the build. Runtime parity is D2's gate. I will verify D1 with a
load-only smoke against the D0 oracle's Inkling-Small GGUF once it is available,
and by building the banded op test.

## Open questions / notes for review

- Dreamcatcher's `llama_kv_cache` is a single struct in `src/llama-context.h`
  rather than a `llama_memory_i` hierarchy. Porting the iswa cache will mean
  either (a) adapting `llama_kv_cache_iswa` to wrap dreamcatcher's struct, or
  (b) introducing the `llama_memory_i` interface. I lean (a) to minimize churn,
  but flag for review.
- The oracle carries a large set of CUDA kernel changes alongside the banded op
  (mmf/mmq/mmvf/mmvq/pad/ssm-conv/argsort). Some may be incidental to the
  banded op. I will port only what the banded op + Inkling graph actually need,
  and note any that are dropped.