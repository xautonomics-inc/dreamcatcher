"""Worked example test matching SPEC-020 and Issue #61 acceptance criteria.

Verifies:
- 46-file library specification (40 transformer blocks, 4 NextN/MTP blocks, embd, output)
- 3-node stage split
- Exact per-node file sets
- Exact per-node byte totals
- Space precheck for all 3 nodes
- Rebalance delta when moving a boundary layer
"""

from __future__ import annotations

from typing import Any

from layer_distribution import (
    compute_blake2b_128,
    files_for_window,
    precheck,
    rebalance,
)


def create_46_file_library_manifest() -> dict[str, Any]:
    """Create a 46-file layer library manifest for a 44-block model with 4 MTP layers."""
    block_size = 850_000_000  # 850 MB per standard transformer block
    nextn_size = 420_000_000  # 420 MB per NextN/MTP block
    embd_size = 650_000_000  # 650 MB token_embd
    output_size = 750_000_000  # 750 MB output_norm + output head

    files = []

    # 40 standard transformer blocks (0..39)
    for i in range(40):
        content = f"simulated-gguf-block-{i}".encode()
        files.append(
            {
                "file": f"blk-{i:05d}.gguf",
                "kind": "layer",
                "abs_index": i,
                "n_bytes_file": block_size,
                "hash": compute_blake2b_128(content),
                "n_tensors": 14,
                "n_bytes_data": block_size - 65536,
            }
        )

    # 4 NextN / MTP blocks (40..43)
    for i in range(40, 44):
        content = f"simulated-gguf-nextn-{i}".encode()
        files.append(
            {
                "file": f"parts-nextn-{i:05d}.gguf",
                "kind": "nextn",
                "abs_index": i,
                "n_bytes_file": nextn_size,
                "hash": compute_blake2b_128(content),
                "n_tensors": 10,
                "n_bytes_data": nextn_size - 65536,
            }
        )

    # parts-embd.gguf (1 file)
    files.append(
        {
            "file": "parts-embd.gguf",
            "kind": "embd",
            "abs_index": None,
            "n_bytes_file": embd_size,
            "hash": compute_blake2b_128(b"simulated-embd-weights"),
            "n_tensors": 1,
            "n_bytes_data": embd_size - 65536,
        }
    )

    # parts-output.gguf (1 file)
    files.append(
        {
            "file": "parts-output.gguf",
            "kind": "output",
            "abs_index": None,
            "n_bytes_file": output_size,
            "hash": compute_blake2b_128(b"simulated-output-weights"),
            "n_tensors": 2,
            "n_bytes_data": output_size - 65536,
        }
    )

    assert len(files) == 46, f"Expected 46 files, got {len(files)}"

    return {
        "format": "gguf-layer-library/v1",
        "created": "2026-09-10T00:00:00Z",
        "tool": "slice_gguf_layers.py",
        "hash_algo": "blake2b-128",
        "source": {
            "glob": "models/model-44b-f16.gguf",
            "shards": [{"name": "model-44b-f16.gguf", "n_bytes": 37_080_000_000}],
            "model_name": "example-44b",
            "arch": "qwen2",
            "block_count": 44,
            "leading_dense_block_count": 0,
            "nextn_predict_layers": 4,
        },
        "files": files,
    }


def test_46_file_library_manifest_shape() -> None:
    """Validate that the manifest has exactly 46 files."""
    manifest = create_46_file_library_manifest()
    assert len(manifest["files"]) == 46

    # Breakdown: 40 layer + 4 nextn + 1 embd + 1 output
    kinds = [f["kind"] for f in manifest["files"]]
    assert kinds.count("layer") == 40
    assert kinds.count("nextn") == 4
    assert kinds.count("embd") == 1
    assert kinds.count("output") == 1


def test_46_file_3_node_split() -> None:
    """Validate 3-node split over the 46-file library."""
    manifest = create_46_file_library_manifest()
    file_sizes = {f["file"]: f["n_bytes_file"] for f in manifest["files"]}

    # Windows:
    # Node 1: [0, 15)  -> 15 layers
    # Node 2: [15, 29) -> 14 layers
    # Node 3: [29, 44) -> 15 layers (11 standard + 4 nextn)
    w1 = files_for_window(manifest, 0, 15)
    w2 = files_for_window(manifest, 15, 29)
    w3 = files_for_window(manifest, 29, 44)

    # Node 1 file count and contents
    assert len(w1) == 17
    assert "parts-embd.gguf" in w1
    assert "parts-output.gguf" in w1
    for i in range(15):
        assert f"blk-{i:05d}.gguf" in w1
    bytes_node1 = sum(file_sizes[f] for f in w1)
    # 15 * 850MB + 650MB + 750MB = 14,150,000,000 bytes
    assert bytes_node1 == 14_150_000_000

    # Node 2 file count and contents
    assert len(w2) == 16
    assert "parts-embd.gguf" in w2
    assert "parts-output.gguf" in w2
    for i in range(15, 29):
        assert f"blk-{i:05d}.gguf" in w2
    bytes_node2 = sum(file_sizes[f] for f in w2)
    # 14 * 850MB + 650MB + 750MB = 13,300,000,000 bytes
    assert bytes_node2 == 13_300_000_000

    # Node 3 file count and contents
    assert len(w3) == 17
    assert "parts-embd.gguf" in w3
    assert "parts-output.gguf" in w3
    for i in range(29, 40):
        assert f"blk-{i:05d}.gguf" in w3
    for i in range(40, 44):
        assert f"parts-nextn-{i:05d}.gguf" in w3
    bytes_node3 = sum(file_sizes[f] for f in w3)
    # 11 * 850MB + 4 * 420MB + 650MB + 750MB = 12,430,000,000 bytes
    assert bytes_node3 == 12_430_000_000

    # Verify prechecks for each node with sufficient space
    p1 = precheck(w1, manifest, free_bytes=20_000_000_000)
    assert p1.feasible is True
    assert p1.required_bytes == 14_150_000_000
    assert p1.surplus_bytes == 5_850_000_000

    p2 = precheck(w2, manifest, free_bytes=15_000_000_000)
    assert p2.feasible is True
    assert p2.required_bytes == 13_300_000_000
    assert p2.surplus_bytes == 1_700_000_000

    p3 = precheck(w3, manifest, free_bytes=10_000_000_000)
    assert p3.feasible is False
    assert p3.required_bytes == 12_430_000_000
    assert p3.deficit_bytes == 2_430_000_000
    assert "shortfall: 2430000000 bytes" in p3.reason


def test_46_file_rebalance() -> None:
    """Validate rebalance delta across nodes in the 46-file library."""
    manifest = create_46_file_library_manifest()

    old_windows = {
        "node1": (0, 15),
        "node2": (15, 29),
        "node3": (29, 44),
    }

    # Shift block 14 from Node 1 to Node 2:
    # Node 1: [0, 14)   (sheds block 14)
    # Node 2: [14, 29)  (adds block 14)
    # Node 3: [29, 44)  (unchanged)
    new_windows = {
        "node1": (0, 14),
        "node2": (14, 29),
        "node3": (29, 44),
    }

    plan = rebalance(old_windows, new_windows, manifest)

    # Node 1
    assert plan["node1"].files_to_add == []
    assert plan["node1"].files_removable == ["blk-00014.gguf"]
    assert plan["node1"].bytes_to_add == 0
    assert plan["node1"].bytes_removable == 850_000_000

    # Node 2
    assert plan["node2"].files_to_add == ["blk-00014.gguf"]
    assert plan["node2"].files_removable == []
    assert plan["node2"].bytes_to_add == 850_000_000
    assert plan["node2"].bytes_removable == 0

    # Node 3
    assert plan["node3"].files_to_add == []
    assert plan["node3"].files_removable == []
    assert plan["node3"].bytes_to_add == 0
    assert plan["node3"].bytes_removable == 0

    # Global rebalance totals
    assert plan.total_bytes_to_add == 850_000_000
    assert plan.total_bytes_removable == 850_000_000
