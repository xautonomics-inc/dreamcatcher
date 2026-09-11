# GGUF Layer-Distribution Engine (`gguf-layer-library/v1`)

Pure, deterministic Python library implementing layer library distribution and validation for multi-node inference stages.

## Specification Overview

In a `gguf-layer-library/v1` deployment, models are stored as discrete per-layer parts:
- `blk-NNNNN.gguf`: Single transformer block per file (absolute index `N`).
- `parts-nextn-NNNNN.gguf`: MTP / NextN speculative prediction block.
- `parts-embd.gguf`: Token embedding weights (`token_embd.weight`).
- `parts-output.gguf`: Output normalization and prediction head (`output_norm.weight`, `output.weight`).
- `parts-other.gguf`: Optional non-block tensors (e.g. `rope_freqs`).
- `manifest.json`: Manifest recording provenance, per-file sizes, and blake2b-128 content hashes.

At launch time, an inference stage loads any arbitrary window `[A, B)` via `--model-dir <dir> --layers A,B`.

## API Functions

### 1. `files_for_window(manifest, a, b)`
Derives the exact file set required by a node executing layer window `[a, b)`:
- Blocks `blk-A..B-1`
- `parts-embd.gguf` and `parts-output.gguf` on **every stage** (loader requirement)
- `parts-nextn-*` when the window covers an MTP layer

### 2. `verify(files, manifest, expected_files=None)`
Verifies file presence, exact byte size, and blake2b-128 cryptographic hash against `manifest.json`:
- Reports per-file status: `pass`, `fail`, `missing`, `size-mismatch`.
- Deterministic and pure: supports in-memory byte maps, precomputed hashes, or filesystem directories.

### 3. `precheck(files, manifest, free_bytes)`
Evaluates whether target storage has sufficient capacity before transferring files:
- Returns `feasible: bool` with exact arithmetic attached (`required_bytes`, `free_bytes`, `surplus_bytes`, `deficit_bytes`).
- Refusals explicitly report the exact byte shortfall.

### 4. `rebalance(old_windows, new_windows, manifest)`
Calculates the migration delta across old and new window allocations per node:
- Returns per-node `files_to_add` and `files_removable`.
- Note: `files_removable` is strictly an advisory report, never an implicit deletion.

## Hash Scopes & Verification Semantics

Manifests declare `hash_scope`:
- `"whole-file"`: `blake2b-128` computed over the raw file bytes.
- `"tensor-aggregate"`: `blake2b-128` computed over concatenated per-tensor digests in write order (legacy slicer format).

### Zero-I/O Scope Discrimination

When `hash_scope` is absent from `manifest.json`, `Manifest.from_dict` deterministically infers the scope without reading model files:
1. If every hashed file's digest reproduces the aggregate of its `tensors[].hash` -> `tensor-aggregate`.
2. If file-level hashes are present but per-tensor hashes are absent -> `whole-file`.
3. If inconsistent or mixed scopes are detected across files -> raises `ValueError` refusing to guess.

### Scope Classification

When whole-file verification (`verifier_scope="whole-file"`) evaluates a `tensor-aggregate` manifest:
- Matching files report `status="unverified-scope-mismatch"`, `passed=True`, `hash_verified=False`.
- Prevents false-positive corruption alerts on intact legacy libraries while safely holding the launch gate until tensor-aware verification is performed.
- `report.scope_verified` requires both `hash_verified=True` and `verified_scope == manifest.hash_scope`.

## Manifest maintenance

Run from this directory with Python 3.10+; no package installation is needed:

```sh
python3 -m layer_distribution.maintenance inventory /data/library
ionice -c 3 python3 -m layer_distribution.maintenance manifest-hash-upgrade /data/library \
  --state /data/integrity-state/library --mib-per-second 64
# Exclude launches and writers, and check consumers on every host first:
python3 -m layer_distribution.maintenance install /data/library \
  --state /data/integrity-state/library --confirmed-idle
```

`prepare` is an alias for `manifest-hash-upgrade`. Preparation streams blake2b-128
hashes at the selected MiB/s limit, increases process niceness by 10, and writes
only to the external state directory. `ionice` is optional on platforms without
it. Budget the **sum** of rates when scheduling more than one scan on shared
storage. Prefer running at the storage source to avoid unnecessary network reads.

Preparation calls the existing `Manifest.from_dict`, `files_for_window`, and
`verify` APIs. The rate-limited reader supplies `(size, digest)` tuples to `verify`
so verification does not start a second, unbounded read. Existing hash mismatches
are failures, never an opportunity to overwrite the expected hash. Missing files,
wrong sizes, extra GGUF files, duplicate or unsafe paths, symlinks, unsupported
algorithms and incomplete layer coverage also prevent preparation.

Every completed file is checkpointed. Repeating the command resumes only entries
with the same library path, manifest bytes and file identity/size/mtime/ctime.
An interrupted file is read again in full. Use `--fresh` for a new physical read
pass: cached digests are **not** a new disk-integrity measurement. Keep the state
directory private and use the same directory for every invocation for that
library; a file lock excludes concurrent checkpoint writers. The checkpoint is
trusted local state, not a signed artifact or an adversarial tamper detector.

`candidate.json` preserves all original fields, including source provenance and
per-tensor metadata. `verification.json` records the before/after results and
file fingerprints. Installation requires an explicit `--confirmed-idle` assertion:
the tool cannot discover remote NFS readers or reserve a distributed launch lease.
The caller must exclude new launches and concurrent writers throughout installation.
It rechecks the manifest, candidate and files, preserves an exact-byte backup
`manifest.json.before-<digest>`, writes a complete sibling, fsyncs it, and atomically
replaces the manifest. Original permissions are retained. Repeating an already
completed installation does not rewrite the manifest. Model files are never edited.

A new hash baseline proves consistency with the bytes read **now**, not that a
previously hashless library survived an earlier incident unchanged. Only hashes
recorded before that incident can establish that comparison. File timestamps do
not detect silent media corruption; run a fresh scan to check that.

### Libraries without a manifest

`manifest-derive` explicitly refuses with exit status 2. This implementation does
not infer completeness or source provenance from filenames. Restore the original
manifest or re-slice a trusted source before upgrading. `inventory` reports this
case as `refused` and `hash_verified: false`. Inventory inconsistencies return
`inventory-failed` and exit status 2, as do refused operations. An inventory reports size and file
membership only; it never claims cryptographic verification.

### Legacy slicer hashes

Some `gguf-layer-library/v1` manifests use `hash_algo: "blake2b-128"` but define
an entry's `hash` as blake2b-128 of the concatenated **binary per-tensor digests**
in write order. This is not a checksum of the GGUF file: it excludes headers and
padding. Counting populated hashes therefore does not establish compatibility
with `verify()`, which checks complete file bytes.

Inventory recognizes entries whose stored hash matches this old convention and
reports `legacy-hash-scope` (exit 2), with the affected filenames. Preparation
refuses these manifests before reading weights. It must not silently replace the
old evidence or interpret the scope mismatch as proof of corrupted storage.
A legacy conversion requires independently checking tensor payloads against their
original hashes, then preserving those hashes while adding whole-file checksums.
That conversion is outside this tool's current scope. Newly prepared manifests
explicitly declare `hash_scope: "whole-file"`; existing source and tensor metadata
is retained.

## Model Slicing (`layer_distribution.slice`)

Slice monolithic or multi-shard GGUF models into per-layer libraries with explicit
`hash_scope: "whole-file"`:

```sh
python3 -m layer_distribution.slice <model-path-or-glob> <out_dir> [--rate MB/s] [--force] [--resume]
```

### Features
- **Library Layout**: Produces `blk-NNNNN.gguf`, `parts-embd.gguf`, `parts-output.gguf`,
  `parts-nextn-NNNNN.gguf`, `parts-other.gguf`, and `manifest.json`.
- **Whole-File Integrity**: Manifest declares `hash_scope: "whole-file"`. `files[].hash` contains
  `blake2b-128` whole-file digests, while `tensors[].hash` retains per-tensor digests for
  granular provenance.
- **Resumable**: With `--resume`, verifies and reuses completed, valid output files without
  re-slicing.
- **Throttling**: Optional `--rate <MB/s>` limits I/O pace to prevent disk contention on shared
  storage.
- **Safety**: Refuses to overwrite non-empty destination directories unless `--force` or
  `--resume` is specified.
- **Sanitized Provenance**: Scrubbed `source.glob` and `source.shards` prevent leaking absolute
  private host paths into publication fixtures.

## Stage Planning (`layer_distribution.plan`)

Compute deterministic stage window allocations across inference nodes based on declared VRAM
and disk capacities, with explicit feasibility arithmetic, assembly flags, and runnable multi-stage launch commands:

```sh
python3 -m layer_distribution.plan <manifest.json> \
  --node gpu-a:24GiB:100GiB:8080:4GiB \
  --node gpu-b:16GiB:80GiB:8081 \
  --model-dir /models/gemma4 \
  [--vram-reserve gpu-a=4GiB] \
  [--json]
```

### Features
- **Deterministic Partitioning**: Allocates contiguous stage windows $[0, B_0), [B_0, B_1), \dots, [B_{n-1}, L)$
  covering all $L$ model layers proportional to node VRAM capacity using largest-remainder integer distribution.
- **Node Capacity Specification**:
  Format: `<name>:<vram>:<disk_free>[:<port>][:<reserve>]` (e.g. `gpu-a:24GiB:100GiB:8080:4GiB` or `gpu-b:16GB:80GB`).
  Supports human byte units (`24GB`, `24GiB`, `500M`).
  Optional `--vram-reserve <name>=<bytes>` flag provides explicit overrides.
- **Feasibility Screen & Arithmetic**:
  - **Disk**: Evaluates `files_for_window()` file sets against available disk space using `precheck()`.
  - **VRAM Feasibility Basis**: Evaluates layer tensor weights plus mandatory loader overhead (`parts-embd.gguf`,
    `parts-output.gguf`, `parts-other.gguf`) against node VRAM capacity. Because base feasibility is weights-only
    (excluding KV cache and compute buffers), reports status as `PASS (weights)` and records `vram_basis`
    in both JSON and reason strings. If an operator configures an explicit reserve, it is added to `vram_required`
    and recorded in the breakdown.
- **Assembly Flags & Launch Commands**:
  - `assembly_flags`: Pure per-layer model loader flag pair (`--model-dir <dir> --layers A,B`).
  - `launch_flags` / `command`: Defaults to runnable `llama-stage-runner` multi-stage pipeline wiring:
    - Head (stage 0): `--role head --connect <next_host>:<next_port> --model-dir <dir> --layers A,B`
    - Relay (intermediate): `--role relay --listen <port> --connect <next_host>:<next_port> --model-dir <dir> --layers A,B`
    - Tail (final stage): `--role tail --listen <port> --model-dir <dir> --layers A,B`
    - Single stage: standalone `--model-dir <dir> --layers A,B`
  - Custom binaries (such as `--binary llama-server`) fall back to server flag rendering.
- **Machine-Readable**: Emits structured JSON with `--json` for integration into orchestrators and console jobs.
- **Zero Dependencies**: Pure Python standard library with no external runtime dependencies.
