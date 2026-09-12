#!/usr/bin/env python3
# Slice a GGUF model into a per-layer "layer library":
#   blk-NNNNN.gguf          one transformer block (abs index NNNNN), tensors relabeled blk.0.*
#   parts-embd.gguf         token_embd.weight
#   parts-output.gguf       output_norm.weight (+ output.weight if the model is untied)
#   parts-nextn-NNNNN.gguf  one NextN/MTP block (abs index NNNNN), tensors relabeled blk.0.*
#   parts-other.gguf        any remaining non-blk tensors (rare; e.g. rope_freqs.weight)
#   manifest.json           provenance + per-file/per-tensor sizes and content hashes
#
# Any stage window [A,B) can then be assembled at LOAD TIME from these files
# (llama-stage-runner --model-dir <dir> --layers A,B), equivalent to a monolithic
#   slice_gguf.py "<glob>" out.gguf A B
# slice of the same window: the loader relabels blk indices window-relative and
# re-derives block_count / leading_dense_block_count / nextn_predict_layers as the
# sum of the per-file values written here.
#
# Mirrors slice_gguf.py semantics: every file keeps the full source KV (self-
# describing, tokenizer included; only GGUF.*/split.* dropped), so any file can
# serve as the assembly's metadata source. Slice-of-slice sources work; abs
# indices are then relative to THAT source.
#
# usage: slice_gguf_layers.py "<model_glob>" <out_dir> [--no-hash] [--force]
import argparse
import glob
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np

import gguf
from gguf import GGUFReader, GGUFWriter, GGUFValueType


def tensor_bytes(t) -> memoryview:
    d = t.data
    if not d.flags["C_CONTIGUOUS"]:
        d = np.ascontiguousarray(d)
    return d.reshape(-1).view(np.uint8).data


def main() -> int:
    ap = argparse.ArgumentParser(description="slice a GGUF model into a per-layer layer library")
    ap.add_argument("model_glob", help="source GGUF file or glob (multi-file sources supported)")
    ap.add_argument("out_dir", help="output directory (the layer library)")
    ap.add_argument("--no-hash", action="store_true", help="skip content hashes in the manifest (faster)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing library dir")
    args = ap.parse_args()

    shards = sorted(glob.glob(args.model_glob))
    assert shards, f"no shards match {args.model_glob}"
    if os.path.isdir(args.out_dir) and os.listdir(args.out_dir) and not args.force:
        sys.exit(f"error: {args.out_dir} exists and is not empty (use --force)")
    os.makedirs(args.out_dir, exist_ok=True)

    meta = GGUFReader(shards[0], "r")
    readers = [meta] + [GGUFReader(s, "r") for s in shards[1:]]
    arch = meta.get_field("general.architecture").contents()

    def srcget(k, default=None):
        f = meta.get_field(f"{arch}.{k}")
        return f.contents() if f else default

    src_blocks  = srcget("block_count")
    src_nextn   = srcget("nextn_predict_layers", 0) or 0
    src_leading = srcget("leading_dense_block_count", 0) or 0
    nextn_lo    = src_blocks - src_nextn        # first NextN block index
    name_field  = meta.get_field("general.name")
    model_name  = name_field.contents() if name_field else os.path.basename(shards[0])
    print(f"[lib] arch={arch} blocks={src_blocks} leading_dense={src_leading} nextn={src_nextn} "
          f"({len(shards)} source shard(s))", flush=True)

    # ---- bucket tensors --------------------------------------------------------
    # buckets: ("layer", i) / ("nextn", i) / ("embd", None) / ("output", None) / ("other", None)
    buckets: dict = {}
    for r in readers:
        for t in r.tensors:
            m = re.match(r"blk\.(\d+)\.(.*)", t.name)
            if m:
                i = int(m.group(1))
                assert i < src_blocks, f"tensor {t.name} beyond block_count={src_blocks}"
                key = ("nextn", i) if i >= nextn_lo else ("layer", i)
                new_name = f"blk.0.{m.group(2)}"
            elif t.name == "token_embd.weight":
                key, new_name = ("embd", None), t.name
            elif t.name in ("output_norm.weight", "output.weight"):
                key, new_name = ("output", None), t.name
            else:
                key, new_name = ("other", None), t.name
            buckets.setdefault(key, []).append((new_name, t))

    for i in range(src_blocks):
        kind = "nextn" if i >= nextn_lo else "layer"
        assert (kind, i) in buckets, f"source has no tensors for blk.{i}"
    others = buckets.get(("other", None), [])
    if others:
        print(f"[lib] WARNING: {len(others)} non-standard tensors -> parts-other.gguf "
              f"(slice_gguf.py DROPS these): {[n for n, _ in others][:8]}", flush=True)

    # ---- write one library file ------------------------------------------------
    # each file keeps full KV; the 3 window-shape keys carry THIS FILE's contribution
    # (layer: block_count=1 etc.; parts: 0) so the loader can sum them per window.
    def write_file(fname: str, kind: str, abs_index, tensors, n_blk, n_dense, n_nextn):
        path = os.path.join(args.out_dir, fname)
        overrides = {
            f"{arch}.block_count":               (n_blk,   GGUFValueType.UINT32),
            f"{arch}.leading_dense_block_count": (n_dense, GGUFValueType.UINT32),
            f"{arch}.nextn_predict_layers":      (n_nextn, GGUFValueType.UINT32),
        }
        w = GGUFWriter(path, arch, endianess=meta.endianess)
        applied = set()
        for f in meta.fields.values():
            if f.name == "general.architecture" or f.name.startswith("GGUF.") or f.name.startswith("split."):
                continue
            vt = f.types[0]
            st = f.types[-1] if vt == GGUFValueType.ARRAY else None
            if f.name in overrides:
                val, vt = overrides[f.name]
                st = None
                applied.add(f.name)
            else:
                val = f.contents()
            if val is not None:
                w.add_key_value(f.name, val, vt, sub_type=st)
        for k, (v, vt) in overrides.items():
            if k not in applied:
                w.add_key_value(k, v, vt)
        for nn, t in tensors:
            w.add_tensor_info(nn, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_ti_data_to_file()
        fhash = hashlib.blake2b(digest_size=16)
        trecs = []
        for nn, t in tensors:
            try:
                w.write_tensor_data(t.data, tensor_endianess=meta.endianess)
            except TypeError:  # older gguf-py (repo-vendored): no per-call endianess override
                w.write_tensor_data(t.data)
            rec = {
                "name": nn,
                "source_name": t.name,
                "shape": [int(x) for x in t.shape],
                "type": t.tensor_type.name,
                "n_bytes": int(t.n_bytes),
            }
            if not args.no_hash:
                th = hashlib.blake2b(tensor_bytes(t), digest_size=16).hexdigest()
                fhash.update(bytes.fromhex(th))
                rec["hash"] = th
            trecs.append(rec)
        w.close()
        entry = {
            "file": fname,
            "kind": kind,
            "abs_index": abs_index,
            "n_tensors": len(tensors),
            "n_bytes_data": int(sum(t.n_bytes for _, t in tensors)),
            "n_bytes_file": os.path.getsize(path),
            "window_kv": {"block_count": n_blk, "leading_dense_block_count": n_dense,
                          "nextn_predict_layers": n_nextn},
            "tensors": trecs,
        }
        if not args.no_hash:
            entry["hash"] = fhash.hexdigest()   # blake2b-128 over the per-tensor hashes, write order
        print(f"[lib] {fname}: {len(tensors)} tensors, {entry['n_bytes_data']/1e6:.1f} MB data", flush=True)
        return entry

    manifest_files = []
    for i in range(nextn_lo):
        manifest_files.append(write_file(
            f"blk-{i:05d}.gguf", "layer", i, buckets[("layer", i)],
            n_blk=1, n_dense=1 if i < src_leading else 0, n_nextn=0))
    for i in range(nextn_lo, src_blocks):
        manifest_files.append(write_file(
            f"parts-nextn-{i:05d}.gguf", "nextn", i, buckets[("nextn", i)],
            n_blk=1, n_dense=0, n_nextn=1))
    if ("embd", None) in buckets:
        manifest_files.append(write_file(
            "parts-embd.gguf", "embd", None, buckets[("embd", None)], 0, 0, 0))
    if ("output", None) in buckets:
        manifest_files.append(write_file(
            "parts-output.gguf", "output", None, buckets[("output", None)], 0, 0, 0))
    if others:
        manifest_files.append(write_file(
            "parts-other.gguf", "other", None, others, 0, 0, 0))

    src_hash = None
    if not args.no_hash:
        # order-independent source content hash: blake2b-128 over sorted (name, tensor-hash)
        acc = hashlib.blake2b(digest_size=16)
        pairs = sorted((t["source_name"], t["hash"]) for e in manifest_files for t in e["tensors"])
        for name, th in pairs:
            acc.update(name.encode())
            acc.update(bytes.fromhex(th))
        src_hash = acc.hexdigest()

    manifest = {
        "format": "gguf-layer-library/v1",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": os.path.basename(__file__),
        "hash_algo": None if args.no_hash else "blake2b-128",
        "source": {
            "glob": args.model_glob,
            "shards": [{"name": os.path.basename(s), "n_bytes": os.path.getsize(s)} for s in shards],
            "model_name": model_name,
            "arch": arch,
            "block_count": int(src_blocks),
            "leading_dense_block_count": int(src_leading),
            "nextn_predict_layers": int(src_nextn),
            "content_hash": src_hash,
        },
        "files": manifest_files,
    }
    mpath = os.path.join(args.out_dir, "manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=1)
    total = sum(e["n_bytes_data"] for e in manifest_files)
    print(f"[lib] DONE {len(manifest_files)} files, {total/1e9:.2f} GB tensor data -> {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
