#!/usr/bin/env python3
"""Prove the manifest on a subset slice by computing papply hashes for selected layer windows.

This script demonstrates that:
1. Different windows produce different papply hashes
2. The same window always produces the same hash (determinism)
3. The hash accounts for window rebasing (includes manifest identity + bounds)
4. NextN/MTP layers are included when the window covers them
"""

from __future__ import annotations

import sys

# Add the tool to the path
sys.path.insert(0, "/home/luna/workspace/dreamcatcher/tools/layer-distribution")

from layer_distribution.engine import compute_blake2b_128, files_for_window, papply_hash
from layer_distribution import Manifest


def make_mock_manifest(
    block_count: int = 40,
    nextn_layers: int = 2,
    dense_blocks: int = 2,
    base_file_size: int = 100_000_000,
    include_other: bool = False,
) -> dict:
    """Construct a realistic mock manifest adhering to gguf-layer-library/v1."""
    files = []
    nextn_lo = block_count - nextn_layers

    # Add standard transformer layers
    for i in range(nextn_lo):
        data = f"layer-{i}-data".encode()
        files.append(
            {
                "file": f"blk-{i:05d}.gguf",
                "kind": "layer",
                "abs_index": i,
                "n_bytes_file": base_file_size + i * 1000,
                "hash": compute_blake2b_128(data),
                "n_tensors": 10,
                "n_bytes_data": base_file_size + i * 1000 - 4096,
            }
        )

    # Add MTP / NextN layers
    for i in range(nextn_lo, block_count):
        data = f"nextn-{i}-data".encode()
        files.append(
            {
                "file": f"parts-nextn-{i:05d}.gguf",
                "kind": "nextn",
                "abs_index": i,
                "n_bytes_file": base_file_size // 2 + i * 1000,
                "hash": compute_blake2b_128(data),
                "n_tensors": 8,
                "n_bytes_data": base_file_size // 2 + i * 1000 - 4096,
            }
        )

    # Add embd
    embd_data = b"token-embedding-weights"
    files.append(
        {
            "file": "parts-embd.gguf",
            "kind": "embd",
            "abs_index": None,
            "n_bytes_file": 50_000_000,
            "hash": compute_blake2b_128(embd_data),
            "n_tensors": 1,
            "n_bytes_data": 50_000_000 - 4096,
        }
    )

    # Add output
    output_data = b"output-norm-and-head-weights"
    files.append(
        {
            "file": "parts-output.gguf",
            "kind": "output",
            "abs_index": None,
            "n_bytes_file": 60_000_000,
            "hash": compute_blake2b_128(output_data),
            "n_tensors": 2,
            "n_bytes_data": 60_000_000 - 4096,
        }
    )

    if include_other:
        other_data = b"rope-freqs-extra-tensors"
        files.append(
            {
                "file": "parts-other.gguf",
                "kind": "other",
                "abs_index": None,
                "n_bytes_file": 10_000,
                "hash": compute_blake2b_128(other_data),
                "n_tensors": 1,
                "n_bytes_data": 5000,
            }
        )

    return {
        "format": "gguf-layer-library/v1",
        "created": "2026-09-10T00:00:00Z",
        "tool": "slice_gguf_layers.py",
        "hash_algo": "blake2b-128",
        "source": {
            "glob": "test-model.gguf",
            "shards": [{"name": "test-model.gguf", "n_bytes": 4_000_000_000}],
            "model_name": "test-model",
            "arch": "llama",
            "block_count": block_count,
            "leading_dense_block_count": dense_blocks,
            "nextn_predict_layers": nextn_layers,
        },
        "files": files,
    }


def main() -> None:
    manifest = make_mock_manifest(block_count=40, nextn_layers=2)
    m = Manifest.from_dict(manifest)

    print("=" * 70)
    print("PROOF: papply_hash on subset slices of a 40-layer library")
    print("=" * 70)
    print(f"Model: {m.source.model_name}, arch={m.source.arch}")
    print(f"block_count={m.source.block_count}, nextn_predict_layers={m.source.nextn_predict_layers}")
    print()

    # Select a handful of layer windows to slice
    slices = [
        (0, 10),    # Start of library
        (10, 20),   # Middle slice
        (20, 30),   # Another middle slice
        (30, 40),   # End of library (covers NextN layers)
        (15, 16),   # Single layer
        (5, 15),    # 10-layer window
    ]

    results = []
    for a, b in slices:
        files = files_for_window(m, a, b)
        h = papply_hash(m, a, b)
        results.append((a, b, h, files))

        print(f"Window [{a:2d}, {b:2d}) — {b - a:2d} layers, {len(files)} files")
        print(f"  papply_hash: {h}")
        print(f"  files: {sorted(files)}")
        print()

    # Prove determinism: same window -> same hash
    print("=" * 70)
    print("DETERMINISM CHECK: same window computed twice")
    print("=" * 70)
    h1 = papply_hash(m, 10, 20)
    h2 = papply_hash(m, 10, 20)
    print(f"  First  computation: {h1}")
    print(f"  Second computation: {h2}")
    assert h1 == h2, "FAIL: papply_hash is not deterministic!"
    print(f"  ✓ Deterministic: hashes match")
    print()

    # Prove uniqueness: different windows -> different hashes
    print("=" * 70)
    print("UNIQUENESS CHECK: all window hashes are distinct")
    print("=" * 70)
    hashes = [r[2] for r in results]
    assert len(set(hashes)) == len(hashes), "FAIL: duplicate hashes found!"
    print(f"  ✓ All {len(hashes)} window hashes are unique")
    print()

    # Prove window rebasing: hash includes manifest identity
    print("=" * 70)
    print("REBASING CHECK: hash changes when manifest identity changes")
    print("=" * 70)
    manifest2 = make_mock_manifest(block_count=40, nextn_layers=2)
    manifest2["source"]["model_name"] = "different-model"
    h_original = papply_hash(m, 10, 20)
    h_different = papply_hash(Manifest.from_dict(manifest2), 10, 20)
    print(f"  Original model hash: {h_original}")
    print(f"  Different model hash: {h_different}")
    assert h_original != h_different, "FAIL: hash should change with manifest identity!"
    print(f"  ✓ Hash changes when manifest identity changes (rebasability proven)")
    print()

    # Prove NextN coverage in end window
    print("=" * 70)
    print("NEXTN COVERAGE: end window includes MTP/NextN layers")
    print("=" * 70)
    end_files = files_for_window(m, 30, 40)
    nextn_files = [f for f in end_files if "nextn" in f]
    print(f"  Window [30, 40) files: {sorted(end_files)}")
    print(f"  NextN files included: {nextn_files}")
    assert len(nextn_files) == 2, "FAIL: expected 2 NextN files in end window!"
    print(f"  ✓ NextN layers correctly included in end window")
    print()

    print("=" * 70)
    print("ALL PROOFS PASSED ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()