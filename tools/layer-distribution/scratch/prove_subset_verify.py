#!/usr/bin/env python3
"""Prove subset slice verification for Inkling models with arch-specific dense_block_count.

Demonstrates that:
1. Slicer correctly preserves 'inkling.dense_block_count' across sliced part files.
2. Window [0, 2) (dense blocks) and window [2, 4) (MoE blocks) each verify cleanly
   using layer_distribution.verify against files_for_window(manifest, a, b).
3. Per-file blake2b-128 cryptographic hashes and byte sizes match the manifest exactly.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Add tools/layer-distribution to Python path
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root / "tools" / "layer-distribution"))

import gguf
import numpy as np
from gguf import GGUFWriter
from layer_distribution import (
    Manifest,
    compute_blake2b_128,
    files_for_window,
    slice_model,
    verify,
)


def create_synthetic_inkling_model(path: Path, n_blocks: int = 4, dense_count: int = 2) -> Path:
    """Create a minimal synthetic Inkling GGUF model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(path, "inkling")
    tensor_dim = 4

    w.add_uint32("inkling.block_count", n_blocks)
    w.add_uint32("inkling.context_length", 64)
    w.add_uint32("inkling.embedding_length", tensor_dim)
    w.add_float32("inkling.attention.layer_norm_rms_epsilon", 1e-5)
    w.add_uint32("inkling.attention.head_count", 1)
    w.add_uint32("inkling.feed_forward_length", tensor_dim)
    w.add_uint32("inkling.dense_block_count", dense_count)
    w.add_uint32("inkling.nextn_predict_layers", 0)
    w.add_string("general.name", "inkling-synthetic-proof")

    n_vocab = 256 + 4
    w.add_tokenizer_model("llama")
    tokens = [f"<0x{i:02X}>" for i in range(256)] + ["P", "i", "n", "g"]
    w.add_token_list(tokens)
    w.add_token_scores([0.0] * n_vocab)
    w.add_token_types([2] * 256 + [1] * 4)
    w.add_bos_token_id(256)
    w.add_eos_token_id(256)
    w.add_unk_token_id(256)

    d_embd = np.arange(n_vocab * tensor_dim, dtype=np.float32).reshape((n_vocab, tensor_dim))
    w.add_tensor_info("token_embd.weight", d_embd.shape, d_embd.dtype, d_embd.nbytes)
    block_tensors = []
    for i in range(n_blocks):
        mat = np.arange(tensor_dim * tensor_dim, dtype=np.float32).reshape((tensor_dim, tensor_dim)) * (i + 2)
        vec = np.ones((tensor_dim,), dtype=np.float32) * (i + 2)
        for name, arr in (
            ("attn_norm", vec), ("attn_q", mat), ("attn_k", mat),
            ("attn_v", mat), ("attn_output", mat), ("ffn_norm", vec),
            ("ffn_gate", mat), ("ffn_down", mat), ("ffn_up", mat),
        ):
            w.add_tensor_info(f"blk.{i}.{name}.weight", arr.shape, arr.dtype, arr.nbytes)
            block_tensors.append(arr)

    d_out = np.arange(tensor_dim, dtype=np.float32) * 10
    w.add_tensor_info("output_norm.weight", d_out.shape, d_out.dtype, d_out.nbytes)
    d_log = np.arange(tensor_dim * n_vocab, dtype=np.float32).reshape((n_vocab, tensor_dim))
    w.add_tensor_info("output.weight", d_log.shape, d_log.dtype, d_log.nbytes)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    w.write_tensor_data(d_embd)
    for t in block_tensors:
        w.write_tensor_data(t)
    w.write_tensor_data(d_out)
    w.write_tensor_data(d_log)
    w.close()
    return path


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        model_path = tmp_path / "model.gguf"
        library_dir = tmp_path / "library"

        print("=" * 80)
        print("1. GENERATING SYNTHETIC INKLING MODEL")
        print("=" * 80)
        create_synthetic_inkling_model(model_path, n_blocks=4, dense_count=2)
        print(f"Source model: {model_path} (4 blocks: 2 dense, 2 MoE)")

        # Verify source metadata
        r_src = gguf.GGUFReader(str(model_path))
        print("Source metadata keys:")
        for k in sorted(r_src.fields.keys()):
            if "dense" in k or "block_count" in k:
                f = r_src.fields[k]
                print(f"  {k} = {f.parts[f.data[0]][0]}")
        print()

        print("=" * 80)
        print("2. SLICING MODEL INTO LAYER LIBRARY")
        print("=" * 80)
        manifest = slice_model(str(model_path), library_dir)
        print(f"Library destination: {library_dir}")
        print(f"Emitted manifest format: {manifest['format']}")
        print(f"Hash algorithm: {manifest['hash_algo']}, scope: {manifest['hash_scope']}")
        print(f"Total files in library: {len(manifest['files'])}")
        print()

        # Check metadata on emitted files
        print("=" * 80)
        print("3. VERIFYING ARCH-SPECIFIC METADATA IN SLICED GGUF FILES")
        print("=" * 80)
        for i in range(4):
            part_name = f"blk-{i:05d}.gguf"
            r = gguf.GGUFReader(str(library_dir / part_name))
            has_dense = "inkling.dense_block_count" in r.fields
            has_lead = "inkling.leading_dense_block_count" in r.fields
            dense_val = r.fields["inkling.dense_block_count"].parts[r.fields["inkling.dense_block_count"].data[0]][0] if has_dense else "N/A"
            blk_val = r.fields["inkling.block_count"].parts[r.fields["inkling.block_count"].data[0]][0]
            print(f"  {part_name}: block_count={blk_val}, dense_block_count={dense_val}, leading_dense_block_count_present={has_lead}")
            assert has_dense, f"Missing inkling.dense_block_count in {part_name}"
            assert not has_lead, f"Stray inkling.leading_dense_block_count in {part_name}"
        print("  ✓ All slices emitted 'inkling.dense_block_count' (zero stray leading_dense keys)")
        print()

        # Demonstrate subset verification for windows
        test_windows = [(0, 2), (2, 4)]
        for a, b in test_windows:
            print("=" * 80)
            print(f"4. SUBSET VERIFICATION FOR WINDOW [{a}, {b}) (layer subset)")
            print("=" * 80)
            subset_files = sorted(files_for_window(manifest, a, b))
            print(f"Window [{a}, {b}) required files ({len(subset_files)} files):")
            for sf in subset_files:
                print(f"  - {sf}")

            report = verify(library_dir, manifest, expected_files=subset_files)
            print()
            print(f"layer_distribution.verify report for window [{a}, {b}):")
            print(f"  passed: {report.passed}")
            print(f"  hash_verified: {report.hash_verified}")
            print(f"  scope_verified: {report.scope_verified}")
            print(f"  verified_scope: {report.verified_scope}")
            print(f"  summary: {report.summary}")
            print()
            print("Per-file verification details:")
            print(f"  {'Filename':<24} {'Status':<8} {'Bytes':<10} {'blake2b-128':<34} {'Match'}")
            print("  " + "-" * 78)
            for fname in subset_files:
                res = report[fname]
                fpath = library_dir / fname
                actual_hash = compute_blake2b_128(fpath.read_bytes())
                match_str = "PASS ✓" if res.passed and res.hash_verified else "FAIL ✗"
                print(f"  {fname:<24} {res.status:<8} {res.actual_size:<10} {res.actual_hash:<34} {match_str}")
                assert res.passed, f"Verification failed for {fname}"
                assert res.hash_verified, f"Hash verification failed for {fname}"
                assert res.actual_hash == actual_hash, f"Hash mismatch for {fname}"
            print()

        print("=" * 80)
        print("ALL SUBSET VERIFICATIONS PASSED ✓")
        print("=" * 80)


if __name__ == "__main__":
    main()
