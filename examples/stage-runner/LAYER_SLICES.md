# Per-layer slices ("layer library") + load-time assembly

Re-balancing the ring today means re-running `slice_gguf.py` per stage window and
redistributing multi-GB monolithic slices. The layer library replaces that with a
ONE-TIME per-layer slicing; any stage window `[A,B)` is then assembled at LOAD TIME
by the stage binary from the per-layer files — equivalent to loading a monolithic
slice of the same window.

## 1. Slicer: `slice_gguf_layers.py`

```
PYTHONPATH=<repo>/gguf-py python3 slice_gguf_layers.py "<model_glob>" <out_dir> [--no-hash] [--force]
```

Reads the source GGUF (multi-file globs supported, like `slice_gguf.py`) and writes:

| file                     | contents                                                | window-shape KV in file |
|--------------------------|---------------------------------------------------------|-------------------------|
| `blk-NNNNN.gguf`         | ONE transformer block (abs index N), tensors `blk.0.*`  | `block_count=1`, `leading_dense_block_count=1` if dense else 0, `nextn=0` |
| `parts-nextn-NNNNN.gguf` | ONE NextN/MTP block (abs index N), tensors `blk.0.*`    | `block_count=1`, `nextn=1` |
| `parts-embd.gguf`        | `token_embd.weight`                                     | all 0 |
| `parts-output.gguf`      | `output_norm.weight` (+ `output.weight` if untied)      | all 0 |
| `parts-other.gguf`       | any other non-blk tensors (rare, e.g. `rope_freqs`) — WARNED; `slice_gguf.py` silently drops these | all 0 |
| `manifest.json`          | provenance, per-file/per-tensor sizes + blake2b-128 hashes | — |

Every file keeps the FULL source KV (tokenizer included, `GGUF.*`/`split.*` dropped)
so each file is self-describing and any of them can serve as the assembly's metadata
source — mirroring `slice_gguf.py`, which copies all KV into every slice.

Slice-of-slice sources work; `abs_index` is then relative to that source.

### manifest.json (format `gguf-layer-library/v1`)

```json
{
  "format": "gguf-layer-library/v1",
  "created": "...", "tool": "slice_gguf_layers.py", "hash_algo": "blake2b-128",
  "source": {
    "glob": "...", "shards": [{"name": "...", "n_bytes": 0}],
    "model_name": "...", "arch": "glm-dsa",
    "block_count": 79, "leading_dense_block_count": 3, "nextn_predict_layers": 1,
    "content_hash": "<blake2b-128 over sorted (source_name, tensor_hash)>"
  },
  "files": [{
    "file": "blk-00000.gguf", "kind": "layer|nextn|embd|output|other",
    "abs_index": 0, "n_tensors": 9, "n_bytes_data": 0, "n_bytes_file": 0,
    "window_kv": {"block_count": 1, "leading_dense_block_count": 1, "nextn_predict_layers": 0},
    "tensors": [{"name": "blk.0.attn_q.weight", "source_name": "blk.0.attn_q.weight",
                 "shape": [..], "type": "Q4_K", "n_bytes": 0, "hash": "..."}],
    "hash": "<blake2b-128 over the per-tensor hashes, write order>"
  }]
}
```

The runner requires `manifest.json` and parses `source.block_count` as a positive
32-bit integer for slicing per-layer metadata arrays. Copy the original manifest
with any partial window of the library. File order and JSON formatting do not
affect this lookup. Missing or invalid source counts are refused before model
loading: part GGUF shape counts (0 or 1) and the highest locally present block
index cannot establish the full source size.

## 2. Loader: `llama_model_load_from_parts()` (libllama)

```c
struct llama_model_part {
    const char * path;
    int32_t blk_base;
    int32_t source_blk_start;
    int32_t source_blk_count;
};
struct llama_model * llama_model_load_from_parts(const struct llama_model_part *, size_t, struct llama_model_params);
```

Backend-agnostic (pure loader level, one mmap/file handle per part, no data copied
twice, monolithic path untouched). Each file's `blk.J` tensors are remapped to
`blk.(blk_base+J)`; `{arch}.block_count`, `{arch}.leading_dense_block_count` and
`{arch}.nextn_predict_layers` are re-derived as the SUM of the per-file values and
injected as internal KV overrides (an explicit user `--override-kv` wins).
Metadata arrays with `source_blk_count` elements are sliced from
`source_blk_start` to the assembled window length. So a window's KV cache and
per-layer hyperparameters both describe the selected blocks, exactly like today's
monolithic slices. Gaps, invalid metadata windows, or out-of-window block indices
abort the load with a clear error.

Debug: `LLAMA_DUMP_TENSOR_HASH=1` logs a FNV-1a-64 hash of every tensor's
in-memory bytes after load (stage-runner also disables repack "extra" bufts under
this env so hashes are byte-comparable across load paths).

## 3. Serving a library in ONE process: `llama-server --model-dir` (recommended)

A layer library is a complete model, not only a pipeline input. `llama-server`
(and `llama-cli`) accept `--model-dir` as an alternative model SOURCE to `-m`:
the enumeration in `common/layer-library.h` composes the part list and the server
loads it through `llama_model_load_from_parts()`. **This is the recommended way to
run a downloaded library** — one process, one API, no ring, no TCP handoff.

```
--model-dir DIR       layer library directory (alternative to -m; mutually exclusive)
--layers A,B          ABSOLUTE window [A,B) into the library; default = the WHOLE model
--stage-parts SPEC    auto (default) | none | comma list of embd,output,nextn,other
```

The default window is the full model (all layers + `parts-embd` + `parts-output` +
any NextN blocks), so a plain `--model-dir DIR` serves exactly what `-m mono.gguf`
would. Every other flag keeps its meaning — `-ngl`, `-ot`, `--tensor-split`, `-fmoe`,
`-cmoe`, `-c`, `-fa`, `--jinja`, sampling, all unchanged:

```
# serve a downloaded library, experts on CPU, everything else on the GPUs
llama-server --model-dir /models/qwen-lib \
  -ngl 999 -cmoe -c 8192 -fa on --jinja --host 127.0.0.1 --port 8080

# equivalent monolithic launch, for comparison
llama-server -m /models/qwen/model-00001-of-00003.gguf \
  -ngl 999 -cmoe -c 8192 -fa on --jinja --host 127.0.0.1 --port 8080
```

A very large CPU-overridden tensor (e.g. a multi-GiB `per_layer_token_embd.weight`
under `-ot '...=CPU'`) no longer aborts the load: the loader now falls back from the
pinned host buffer to the plain CPU buffer for that tensor and logs one line.
`LLAMA_NO_HOST_OVERRIDES=1` remains the explicit all-tensors override.

Smoke it with `ci/smoke-serve.sh http://127.0.0.1:8080 --min-tps 1.0`, which screens
the completion for degenerate output as well as throughput.

## 4. Stage binary: launch parameters

WIRED in `stage-runner.cpp`, over the same `common/layer-library.h` enumeration the
server uses: `--model-dir` / `--layers` / `--stage-parts` select and assemble the
part files exactly as designed below; stage launches may use `--model-dir` in place
of `-m` slices. Deviations in the ik v1 runner (see its header comment): no
`--pipeline` (depth==slots, non-pipelined v1), and its roles are
`server` / `head` / `tail` / `relay` only. There is no `gen` role and no
single-process generation mode here — for that, use `llama-server --model-dir`
(section 3). `STAGE_EMIT=hidden` file emit is kept.

```
--model-dir DIR       layer library directory (alternative to -m; mutually exclusive)
--layers A,B          ABSOLUTE window [A,B) into the library; default = full model
--stage-parts SPEC    auto (default) | none | comma list of embd,output,nextn,other
```

`auto` mirrors today's monolithic slices exactly: `parts-embd` + `parts-output` are
included in EVERY slice (the graph code requires `token_embd` present, and
`slice_gguf.py` ships embd+output in all slices), NextN blocks are included iff the
window covers their indices (which also makes the summed `nextn_predict_layers` KV
match `slice_gguf.py`'s tail-keeps-MTP / middle-gets-0 behavior).

`STAGE_IL_START/STAGE_IL_END` are NOT reused for file selection: the live launch
convention (see `ring-ops/*.sh`) sets them SLICE-RELATIVE (e.g. the `[60,61)` stage
loads `q4ks-60-61.gguf` with `STAGE_IL_START=0 STAGE_IL_END=1`), so overloading them
with absolute meaning would silently mis-assemble. Keep setting them exactly as
today; only the model source changes:

```
# head [0,19)   (was: -m q4ks-0-19.gguf)
STAGE_ACTIVE=1 STAGE_IL_START=0 STAGE_IL_END=19 STAGE_EMIT=hidden \
  llama-stage-runner --model-dir /models/glm52-lib --layers 0,19 \
    -ngl 999 --split-mode tensor --role head --connect NEXT:PORT --pipeline

# middle/relay [60,61)   (was: -m q4ks-60-61.gguf)
STAGE_ACTIVE=1 STAGE_IL_START=0 STAGE_IL_END=1 STAGE_EMIT=hidden \
  llama-stage-runner --model-dir /models/glm52-lib --layers 60,61 \
    -ngl 999 --split-mode tensor --role relay --listen 53550 --connect NEXT:PORT --pipeline

# tail [61,79)   (window covers the NextN block -> MTP kept automatically)
STAGE_ACTIVE=1 STAGE_IL_START=0 STAGE_IL_END=18 \
  llama-stage-runner --model-dir /models/glm52-lib --layers 61,79 \
    -ngl 999 --split-mode tensor --role tail --listen 53600 --token-return HEAD:PORT --pipeline
```

Local correctness harness (used for the equivalence proof):

```
# greedy generation (temp 0, top_k 1, fixed seed) — identical token streams across
# load paths. Run each server in turn on the same port and diff the completions.
llama-server -m model.gguf            -c 8192 -fa on --port 8080   # then POST /completion
llama-server --model-dir lib          -c 8192 -fa on --port 8080   # same body, same tokens

# loader-level byte equality for any window (either binary)
LLAMA_DUMP_TENSOR_HASH=1 llama-stage-runner -m mono-5-9.gguf ... | grep tensor-hash | sort
LLAMA_DUMP_TENSOR_HASH=1 llama-stage-runner --model-dir lib --layers 5,9 ... | grep tensor-hash | sort
```

> `llama-stage-runner --role gen --prompt … --last` is NOT a generation path: the
> runner's roles are `server` / `head` / `tail` / `relay`, and the FILE-mode emit path
> forces embeddings mode (`STAGE_EMIT=hidden`), so it cannot produce logits to sample.
> The wider stage-runner documentation drift is tracked separately (issue #89).

## Verified (TinyLlama-1.1B Q4_K_M, CPU build, 2026-07-12)

- window [5,9), [17,22), and full [0,22) vs the ORIGINAL gguf: all in-memory
  tensor hashes identical (39/39, 48/48, 201/201);
- temp-0 generation over the assembled full model: 48/48 tokens identical to the
  monolithic file;
- STAGE_ACTIVE relative-window hidden-state emit ([5,9) slice, 4-layer stage):
  output file byte-identical monolithic vs assembled;
- `--no-mmap` assembly path works; monolithic `-m` path untouched.

## Re-verified after rebase (TinyLlama-1.1B Q4_K_M, CUDA build, 2026-09-02)

Branch rebased onto stage-runner rebase (upstream-HEAD lineage); re-proved with
a loader-level harness (`check54`: `llama_model_load_from_parts` over explicit
`path:blk_base` parts) on a GGML_CUDA=ON build (sm_120):

- slicer regenerates the 24-file library from the monolithic gguf (0.67 GB);
- full-window assembly [0,22) = 24 parts (embd + blk 0..21 + output):
  `block_count` override applied, `n_layer=22`, load OK;
- 201/201 in-memory tensor hashes IDENTICAL assembly vs monolithic `-m` load;
- window [0,5) assembly (7 parts: embd + blk 0..4 + output) loads:
  `n_layer=5`, 48 tensors;
- gap in the window (blk 0,1,3): load aborts — `invalid assembly: no tensors
  for blk.2 (window block_count=3)`.

## Window-relative vs source-absolute per-layer values

A window is a *slice of a source model*, not a small model of its own. Anything
the source model indexes by its **absolute** block number has to be rebuilt from
the window's offset, or a window that does not start at block 0 reads the wrong
value for every one of its layers. The loader therefore records the window it
assembled and hands it to the rest of the loader:

- `llama_model_loader::window_il_offset` / `window_n_layer_src` — set from the
  `llama_model_part` window; `0` / `block_count` for a monolithic `-m` file.
- `llama_hparams::il_offset` / `n_layer_src`, read through `il_abs(il)` and
  `n_layer_source()`. For a monolithic model `il_abs()` is the identity, so
  nothing about the `-m` path changes.

Two families of value need it:

1. **"every k-th block" patterns** — the linear-vs-full attention interval
   (`full_attention_interval`) and sliding-window periods. Built from the local
   index, a window starting at a block that is not a multiple of the period
   gives every one of its layers the wrong kind of attention. A window whose
   start *is* a multiple of the period was accidentally correct, which is why
   24- and 40-block splits of a period-4 model never showed it.
2. **Per-layer token-embedding tables** — `per_layer_token_embd` is stored whole,
   one slice per source block, and is indexed by the source block number. It is
   sized by `n_layer_source()`, and the window's slice
   `[il_offset, il_offset + n_layer)` is cut out at graph-build time.

### Per-layer embeddings need the batch's TOKEN IDS

A per-layer embedding is a lookup keyed by the token, so a window that owns such
a block needs the token ids as well as the table. A pipeline stage past the first
is handed **hidden states**, not tokens, and the lookup silently falls back to a
placeholder row. Two architectures are affected, differently:

- **gemma4** with `embedding_length_per_layer_input > 0`: *every* block consumes
  a per-layer embedding, so only the stage that owns source block 0 can be
  correct. A later window now warns at load time. (Before this change such a
  window could not load at all: the table was sized with the window's block
  count and did not match the stored shape.)
- **qwen4exp**: only the blocks listed in `ple.layers` consume the PLE n-gram
  table — a single block in the shipped Flash-Next models. `ple.layers` is
  rebased onto the window and dropped when empty, so a window with no PLE block
  correctly does **not** load the table. The loader now says that explicitly
  instead of leaving only a bare `skipped N unused tensors` line, which reads
  like a window losing a weight it needed. A window that *does* own a PLE block
  and does not start at block 0 warns.

Carrying token ids alongside the hidden handoff would lift the restriction; the
transport does not do that today.

## Sampling parity with a monolithic server

A tail samples the **last row of each sequence** and discards the rest. It used
to ask `llama_decode` for logits on *every* row of the handoff, which makes the
final `lm_head` an `n_rows`-wide matmul where a monolithic server runs a
one-row one — a different kernel, so different rounding, so a different argmax
whenever the top two candidates are close. The tail now requests logits only on
the rows it reads. On an 80B-A6B `qwen4exp` this changed the **first** sampled
token of a 7-token prompt, and made two different splits of the same model agree
with each other where they had not before.
