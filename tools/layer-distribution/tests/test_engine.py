"""Exhaustive unit test suite for the GGUF layer-distribution engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from layer_distribution import (
    FileStatus,
    HashScope,
    Manifest,
    compute_blake2b_128,
    files_for_window,
    precheck,
    rebalance,
    verify,
)


def make_mock_manifest(
    block_count: int = 40,
    nextn_layers: int = 2,
    dense_blocks: int = 2,
    base_file_size: int = 100_000_000,
    include_other: bool = False,
) -> dict[str, Any]:
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


# ============================================================================
# 1. files_for_window tests
# ============================================================================


def test_files_for_window_start() -> None:
    """Test window at the start of the library [0, 10)."""
    manifest = make_mock_manifest(block_count=40, nextn_layers=2)
    files = files_for_window(manifest, 0, 10)

    # Must contain blk-00000..blk-00009
    expected_blks = {f"blk-{i:05d}.gguf" for i in range(10)}
    assert expected_blks.issubset(files)

    # Must contain parts-embd and parts-output on EVERY stage
    assert "parts-embd.gguf" in files
    assert "parts-output.gguf" in files

    # Must NOT contain other blocks or nextn blocks
    assert "blk-00010.gguf" not in files
    assert "parts-nextn-00038.gguf" not in files
    assert len(files) == 10 + 2


def test_files_for_window_end() -> None:
    """Test window at the end of the library [30, 40) covering NextN layers."""
    manifest = make_mock_manifest(block_count=40, nextn_layers=2)
    files = files_for_window(manifest, 30, 40)

    # Blocks 30..37 are standard layers
    for i in range(30, 38):
        assert f"blk-{i:05d}.gguf" in files

    # Blocks 38..39 are NextN/MTP layers
    assert "parts-nextn-00038.gguf" in files
    assert "parts-nextn-00039.gguf" in files

    # Every stage must have embd and output
    assert "parts-embd.gguf" in files
    assert "parts-output.gguf" in files
    assert len(files) == 8 + 2 + 2


def test_files_for_window_single_layer() -> None:
    """Test single-layer window [15, 16)."""
    manifest = make_mock_manifest(block_count=40, nextn_layers=2)
    files = files_for_window(manifest, 15, 16)

    assert files == {"blk-00015.gguf", "parts-embd.gguf", "parts-output.gguf"}
    assert len(files) == 3


def test_files_for_window_covering_mtp() -> None:
    """Test window strictly covering MTP layer [38, 39)."""
    manifest = make_mock_manifest(block_count=40, nextn_layers=2)
    files = files_for_window(manifest, 38, 39)

    assert files == {"parts-nextn-00038.gguf", "parts-embd.gguf", "parts-output.gguf"}
    assert len(files) == 3


def test_files_for_window_includes_other() -> None:
    """Test that parts-other.gguf is included when present in manifest."""
    manifest = make_mock_manifest(block_count=10, nextn_layers=0, include_other=True)
    files = files_for_window(manifest, 0, 5)

    assert "parts-other.gguf" in files
    assert "parts-embd.gguf" in files
    assert "parts-output.gguf" in files


def test_files_for_window_invalid_bounds() -> None:
    """Test error handling for invalid windows."""
    manifest = make_mock_manifest(block_count=40)

    with pytest.raises(ValueError, match="Invalid layer window"):
        files_for_window(manifest, -1, 10)

    with pytest.raises(ValueError, match="Invalid layer window"):
        files_for_window(manifest, 10, 5)

    with pytest.raises(ValueError, match="Invalid layer window"):
        files_for_window(manifest, 5, 5)

    with pytest.raises(ValueError, match="Invalid layer window"):
        files_for_window(manifest, 0, 41)


# ============================================================================
# 2. verify tests
# ============================================================================


def test_verify_all_pass_in_memory() -> None:
    """Test clean verification pass with pure in-memory byte mapping."""
    manifest = make_mock_manifest(block_count=4, nextn_layers=0)
    file_map = {
        "blk-00000.gguf": b"layer-0-data",
        "blk-00001.gguf": b"layer-1-data",
        "blk-00002.gguf": b"layer-2-data",
        "blk-00003.gguf": b"layer-3-data",
        "parts-embd.gguf": b"token-embedding-weights",
        "parts-output.gguf": b"output-norm-and-head-weights",
    }
    for mf in manifest["files"]:
        mf["n_bytes_file"] = len(file_map[mf["file"]])

    report = verify(file_map, manifest)
    assert report.passed is True
    assert report.summary[FileStatus.PASS.value] == 6
    assert report.summary[FileStatus.FAIL.value] == 0
    assert report.summary[FileStatus.MISSING.value] == 0
    assert report.summary[FileStatus.SIZE_MISMATCH.value] == 0

    for fname in file_map:
        assert report[fname].status == FileStatus.PASS.value
        assert report[fname].passed is True


def test_verify_missing_blk_file() -> None:
    """Test detection of missing blk file."""
    manifest = make_mock_manifest(block_count=3, nextn_layers=0)
    file_map = {
        "blk-00000.gguf": b"layer-0-data",
        # blk-00001.gguf is intentionally omitted
        "blk-00002.gguf": b"layer-2-data",
        "parts-embd.gguf": b"token-embedding-weights",
        "parts-output.gguf": b"output-norm-and-head-weights",
    }
    for mf in manifest["files"]:
        if mf["file"] in file_map:
            mf["n_bytes_file"] = len(file_map[mf["file"]])

    report = verify(file_map, manifest)
    assert report.passed is False
    assert report.summary[FileStatus.MISSING.value] == 1
    assert report["blk-00001.gguf"].status == FileStatus.MISSING.value
    assert report["blk-00001.gguf"].passed is False


def test_verify_hash_mismatch() -> None:
    """Test detection of corrupted content (hash mismatch with matching size)."""
    manifest = make_mock_manifest(block_count=2, nextn_layers=0)
    corrupted_data = b"layer-0-bad!"  # same length as b"layer-0-data" (12 bytes)
    assert len(corrupted_data) == len(b"layer-0-data")

    file_map = {
        "blk-00000.gguf": corrupted_data,
        "blk-00001.gguf": b"layer-1-data",
        "parts-embd.gguf": b"token-embedding-weights",
        "parts-output.gguf": b"output-norm-and-head-weights",
    }
    for mf in manifest["files"]:
        mf["n_bytes_file"] = len(file_map[mf["file"]])

    report = verify(file_map, manifest)
    assert report.passed is False
    assert report.summary[FileStatus.FAIL.value] == 1
    assert report["blk-00000.gguf"].status == FileStatus.FAIL.value
    assert "Hash mismatch" in report["blk-00000.gguf"].detail


def test_verify_size_mismatch() -> None:
    """Test detection of truncated or padded file (size mismatch)."""
    manifest = make_mock_manifest(block_count=2, nextn_layers=0)
    truncated_data = b"layer-0"  # shorter than 12 bytes

    file_map = {
        "blk-00000.gguf": truncated_data,
        "blk-00001.gguf": b"layer-1-data",
        "parts-embd.gguf": b"token-embedding-weights",
        "parts-output.gguf": b"output-norm-and-head-weights",
    }
    for mf in manifest["files"]:
        if mf["file"] == "blk-00000.gguf":
            mf["n_bytes_file"] = 12  # expected 12, but actual is 7
        else:
            mf["n_bytes_file"] = len(file_map[mf["file"]])

    report = verify(file_map, manifest)
    assert report.passed is False
    assert report.summary[FileStatus.SIZE_MISMATCH.value] == 1
    assert report["blk-00000.gguf"].status == FileStatus.SIZE_MISMATCH.value
    assert "Size mismatch" in report["blk-00000.gguf"].detail


def test_verify_with_filesystem_tmp_path(tmp_path: Path) -> None:
    """Test filesystem verification using real files in tmp_path fixture."""
    manifest = make_mock_manifest(block_count=2, nextn_layers=0)

    (tmp_path / "blk-00000.gguf").write_bytes(b"layer-0-data")
    (tmp_path / "blk-00001.gguf").write_bytes(b"layer-1-data")
    (tmp_path / "parts-embd.gguf").write_bytes(b"token-embedding-weights")
    (tmp_path / "parts-output.gguf").write_bytes(b"output-norm-and-head-weights")

    for mf in manifest["files"]:
        mf["n_bytes_file"] = (tmp_path / mf["file"]).stat().st_size

    # Verify passing dir
    report = verify(tmp_path, manifest)
    assert report.passed is True
    assert report.summary[FileStatus.PASS.value] == 4

    # Remove one file -> missing
    (tmp_path / "blk-00001.gguf").unlink()
    report_missing = verify(tmp_path, manifest)
    assert report_missing.passed is False
    assert report_missing.summary[FileStatus.MISSING.value] == 1


# ============================================================================
# 3. precheck tests
# ============================================================================


def test_precheck_exactly_fits() -> None:
    """Test precheck when free space exactly matches required bytes."""
    manifest = make_mock_manifest(block_count=5, nextn_layers=0)
    files = files_for_window(manifest, 0, 3)

    file_size_map = {mf["file"]: mf["n_bytes_file"] for mf in manifest["files"]}
    required_bytes = sum(file_size_map[f] for f in files)

    res = precheck(files, manifest, free_bytes=required_bytes)
    assert res.feasible is True
    assert bool(res) is True
    assert res.required_bytes == required_bytes
    assert res.free_bytes == required_bytes
    assert res.deficit_bytes == 0
    assert res.surplus_bytes == 0
    assert "Feasible" in res.reason
    assert "surplus: 0 bytes" in res.reason


def test_precheck_one_byte_short() -> None:
    """Test precheck when free space is exactly one byte short of required."""
    manifest = make_mock_manifest(block_count=5, nextn_layers=0)
    files = files_for_window(manifest, 0, 3)

    file_size_map = {mf["file"]: mf["n_bytes_file"] for mf in manifest["files"]}
    required_bytes = sum(file_size_map[f] for f in files)

    res = precheck(files, manifest, free_bytes=required_bytes - 1)
    assert res.feasible is False
    assert bool(res) is False
    assert res.required_bytes == required_bytes
    assert res.free_bytes == required_bytes - 1
    assert res.deficit_bytes == 1
    assert res.surplus_bytes == 0
    assert "Infeasible" in res.reason
    assert "shortfall: 1 bytes" in res.reason


def test_precheck_generous_free_space() -> None:
    """Test precheck with plenty of free disk space."""
    manifest = make_mock_manifest(block_count=5, nextn_layers=0)
    files = files_for_window(manifest, 0, 2)
    file_size_map = {mf["file"]: mf["n_bytes_file"] for mf in manifest["files"]}
    required_bytes = sum(file_size_map[f] for f in files)

    res = precheck(files, manifest, free_bytes=required_bytes + 50_000_000)
    assert res.feasible is True
    assert res.surplus_bytes == 50_000_000
    assert res.deficit_bytes == 0


def test_precheck_negative_free_bytes() -> None:
    """Test error when negative free bytes are provided."""
    manifest = make_mock_manifest(block_count=5)
    with pytest.raises(ValueError, match="cannot be negative"):
        precheck(["parts-embd.gguf"], manifest, free_bytes=-10)


# ============================================================================
# 4. rebalance tests
# ============================================================================


def test_rebalance_overlapping_windows() -> None:
    """Test rebalance delta computation when windows overlap."""
    manifest = make_mock_manifest(block_count=40, nextn_layers=2)

    old_windows = {
        "node1": (0, 15),
        "node2": (15, 30),
        "node3": (30, 40),
    }

    new_windows = {
        "node1": (0, 20),
        "node2": (20, 30),
        "node3": (30, 40),
    }

    plan = rebalance(old_windows, new_windows, manifest)

    # Node 1
    d1 = plan["node1"]
    assert d1.old_window == (0, 15)
    assert d1.new_window == (0, 20)
    assert d1.files_to_add == [f"blk-{i:05d}.gguf" for i in range(15, 20)]
    assert d1.files_removable == []
    assert d1.bytes_to_add > 0
    assert d1.bytes_removable == 0

    # Node 2
    d2 = plan["node2"]
    assert d2.old_window == (15, 30)
    assert d2.new_window == (20, 30)
    assert d2.files_to_add == []
    assert d2.files_removable == [f"blk-{i:05d}.gguf" for i in range(15, 20)]
    assert d2.bytes_to_add == 0
    assert d2.bytes_removable > 0
    assert d1.bytes_to_add == d2.bytes_removable

    # Node 3: unchanged
    d3 = plan["node3"]
    assert d3.old_window == (30, 40)
    assert d3.new_window == (30, 40)
    assert d3.files_to_add == []
    assert d3.files_removable == []
    assert d3.bytes_to_add == 0
    assert d3.bytes_removable == 0

    # Fleet totals
    assert plan.total_bytes_to_add == d1.bytes_to_add
    assert plan.total_bytes_removable == d2.bytes_removable


def test_rebalance_add_and_remove_nodes() -> None:
    """Test rebalance when a node is decommissioned and a new node joins."""
    manifest = make_mock_manifest(block_count=20, nextn_layers=0)

    old_windows = {
        "nodeA": (0, 10),
        "nodeB": (10, 20),
    }
    new_windows = {
        "nodeA": (0, 10),
        "nodeC": (10, 20),
    }

    plan = rebalance(old_windows, new_windows, manifest)

    d_b = plan["nodeB"]
    assert d_b.new_window is None
    assert d_b.files_to_add == []
    assert len(d_b.files_removable) == 10 + 2  # 10 blks + embd + output

    d_c = plan["nodeC"]
    assert d_c.old_window is None
    assert len(d_c.files_to_add) == 10 + 2
    assert d_c.files_removable == []


def test_verify_hashless_manifest_right_size_wrong_content() -> None:
    """Test verification against a hashless manifest (e.g. hash_algo: null).

    Right-size / wrong-content files report 'unverified-no-hash' and passed=True,
    but report-level hash_verified is False so callers can enforce cryptographic gating.
    """
    manifest = make_mock_manifest(block_count=3, nextn_layers=0)
    manifest["hash_algo"] = None
    for mf in manifest["files"]:
        mf["hash"] = None

    # Files with right size (e.g. 12 bytes) but completely arbitrary/wrong content
    corrupt_file_map = {
        "blk-00000.gguf": b"123456789012",
        "blk-00001.gguf": b"abcdefghijkl",
        "blk-00002.gguf": b"XXXXXXXXXXXX",
        "parts-embd.gguf": b"token-embedding-weights",
        "parts-output.gguf": b"output-norm-and-head-weights",
    }
    for mf in manifest["files"]:
        mf["n_bytes_file"] = len(corrupt_file_map[mf["file"]])

    report = verify(corrupt_file_map, manifest)
    # passed=True because sizes match and nothing is detectably corrupted
    assert report.passed is True
    # hash_verified=False because manifest carried no hashes
    assert report.hash_verified is False
    assert report.summary[FileStatus.UNVERIFIED_NO_HASH.value] == 5
    assert report.summary[FileStatus.PASS.value] == 0
    assert report.summary[FileStatus.FAIL.value] == 0

    for fname in corrupt_file_map:
        res = report[fname]
        assert res.status == FileStatus.UNVERIFIED_NO_HASH.value
        assert res.passed is True
        assert res.hash_verified is False
        assert res.expected_hash is None
        assert "manifest has no hash" in res.detail


def test_verify_hashless_manifest_with_size_mismatch_and_missing() -> None:
    """Test that size-changing corruption or missing files fail even without hashes."""
    manifest = make_mock_manifest(block_count=3, nextn_layers=0)
    manifest["hash_algo"] = None
    for mf in manifest["files"]:
        mf["hash"] = None

    file_map = {
        "blk-00000.gguf": b"short",  # size mismatch
        # blk-00001.gguf is missing
        "blk-00002.gguf": b"123456789012",
        "parts-embd.gguf": b"token-embedding-weights",
        "parts-output.gguf": b"output-norm-and-head-weights",
    }
    for mf in manifest["files"]:
        if mf["file"] == "blk-00000.gguf":
            mf["n_bytes_file"] = 12
        elif mf["file"] in file_map:
            mf["n_bytes_file"] = len(file_map[mf["file"]])

    report = verify(file_map, manifest)
    assert report.passed is False
    assert report.hash_verified is False
    assert report.summary[FileStatus.SIZE_MISMATCH.value] == 1
    assert report.summary[FileStatus.MISSING.value] == 1
    assert report.summary[FileStatus.UNVERIFIED_NO_HASH.value] == 3


def test_enrich_manifest_hashes() -> None:
    """Test upgrading a hashless manifest by computing blake2b-128 checksums."""
    from layer_distribution import enrich_manifest_hashes

    manifest = make_mock_manifest(block_count=2, nextn_layers=0)
    for mf in manifest["files"]:
        mf["hash"] = None
    manifest["hash_algo"] = None

    file_map = {
        "blk-00000.gguf": b"block-0-data",
        "blk-00001.gguf": b"block-1-data",
        "parts-embd.gguf": b"embd-data",
        "parts-output.gguf": b"output-data",
    }
    for mf in manifest["files"]:
        mf["n_bytes_file"] = len(file_map[mf["file"]])

    # Initially hashless
    initial_report = verify(file_map, manifest)
    assert initial_report.passed is True
    assert initial_report.hash_verified is False

    # Upgrade manifest
    enriched = enrich_manifest_hashes(manifest, file_map)
    assert enriched.hash_algo == "blake2b-128"
    for mf in enriched.files:
        assert mf.hash is not None
        assert len(mf.hash) == 32

    # Now verification against enriched manifest is fully hash-verified
    upgraded_report = verify(file_map, enriched)
    assert upgraded_report.passed is True
    assert upgraded_report.hash_verified is True
    assert upgraded_report.scope_verified is True
    assert upgraded_report.hash_scope == "whole-file"
    assert upgraded_report.verified_scope == "whole-file"
    assert upgraded_report.summary[FileStatus.PASS.value] == 4
    assert upgraded_report.summary[FileStatus.UNVERIFIED_NO_HASH.value] == 0


def test_tensor_aggregate_scope_inference() -> None:
    """Test deterministic zero-I/O inference of tensor-aggregate hash scope."""
    import hashlib

    # Create files with tensor-level hashes whose aggregate equals file hash
    t1_hash = hashlib.blake2b(b"tensor-1-bytes", digest_size=16).hexdigest()
    t2_hash = hashlib.blake2b(b"tensor-2-bytes", digest_size=16).hexdigest()
    agg = hashlib.blake2b(digest_size=16)
    agg.update(bytes.fromhex(t1_hash))
    agg.update(bytes.fromhex(t2_hash))
    file_agg_hash = agg.hexdigest()

    manifest_dict: dict[str, Any] = {
        "format": "gguf-layer-library/v1",
        "hash_algo": "blake2b-128",
        # hash_scope intentionally omitted to test inference
        "source": {"model_name": "test", "arch": "llama", "block_count": 1},
        "files": [
            {
                "file": "blk-00000.gguf",
                "kind": "layer",
                "abs_index": 0,
                "n_bytes_file": 2048,
                "hash": file_agg_hash,
                "tensors": [
                    {"name": "blk.0.attn_q.weight", "hash": t1_hash},
                    {"name": "blk.0.attn_k.weight", "hash": t2_hash},
                ],
            },
            {
                "file": "parts-embd.gguf",
                "kind": "embd",
                "abs_index": None,
                "n_bytes_file": 1024,
                "hash": file_agg_hash,
                "tensors": [
                    {"name": "token_embd.weight", "hash": t1_hash},
                    {"name": "output_norm.weight", "hash": t2_hash},
                ],
            },
            {
                "file": "parts-output.gguf",
                "kind": "output",
                "abs_index": None,
                "n_bytes_file": 1024,
                "hash": file_agg_hash,
                "tensors": [
                    {"name": "output.weight", "hash": t1_hash},
                    {"name": "norm.weight", "hash": t2_hash},
                ],
            },
        ],
    }

    manifest = Manifest.from_dict(manifest_dict)
    assert manifest.hash_scope == HashScope.TENSOR_AGGREGATE.value


def test_whole_file_scope_inference_no_per_tensor_hashes() -> None:
    """Test zero-I/O inference of whole-file hash scope when per-tensor hashes are absent."""
    manifest_dict: dict[str, Any] = {
        "format": "gguf-layer-library/v1",
        "hash_algo": "blake2b-128",
        # hash_scope omitted
        "source": {"model_name": "test", "arch": "llama", "block_count": 1},
        "files": [
            {
                "file": "blk-00000.gguf",
                "kind": "layer",
                "abs_index": 0,
                "n_bytes_file": 2048,
                "hash": "0123456789abcdef0123456789abcdef",
                # tensors present but contain no hashes (like Q4 after upgrade)
                "tensors": [
                    {"name": "blk.0.attn_q.weight", "type": "Q4_0", "n_bytes": 1024},
                    {"name": "blk.0.attn_k.weight", "type": "Q4_0", "n_bytes": 1024},
                ],
            },
            {
                "file": "parts-embd.gguf",
                "kind": "embd",
                "abs_index": None,
                "n_bytes_file": 1024,
                "hash": "0123456789abcdef0123456789abcdef",
                "tensors": [],
            },
            {
                "file": "parts-output.gguf",
                "kind": "output",
                "abs_index": None,
                "n_bytes_file": 1024,
                "hash": "0123456789abcdef0123456789abcdef",
            },
        ],
    }

    manifest = Manifest.from_dict(manifest_dict)
    assert manifest.hash_scope == HashScope.WHOLE_FILE.value


def test_explicit_hash_scope_override() -> None:
    """Explicit hash_scope in manifest overrides any inference."""
    manifest_dict: dict[str, Any] = {
        "format": "gguf-layer-library/v1",
        "hash_algo": "blake2b-128",
        "hash_scope": "whole-file",
        "source": {"model_name": "test", "arch": "llama", "block_count": 1},
        "files": [
            {
                "file": "blk-00000.gguf",
                "kind": "layer",
                "abs_index": 0,
                "n_bytes_file": 2048,
                "hash": "abcdefabcdefabcdefabcdefabcdefab",
            }
        ],
    }
    m = Manifest.from_dict(manifest_dict)
    assert m.hash_scope == "whole-file"

    manifest_dict["hash_scope"] = "tensor-aggregate"
    m2 = Manifest.from_dict(manifest_dict)
    assert m2.hash_scope == "tensor-aggregate"


def test_mixed_hash_scope_refusal() -> None:
    """Inconsistent hash scopes across files in one library raises ValueError."""
    import hashlib

    t_hash = hashlib.blake2b(b"tensor", digest_size=16).hexdigest()
    agg = hashlib.blake2b(digest_size=16)
    agg.update(bytes.fromhex(t_hash))
    file_agg = agg.hexdigest()

    manifest_dict: dict[str, Any] = {
        "format": "gguf-layer-library/v1",
        "hash_algo": "blake2b-128",
        "source": {"model_name": "test", "arch": "llama", "block_count": 2},
        "files": [
            {
                "file": "blk-00000.gguf",
                "kind": "layer",
                "abs_index": 0,
                "n_bytes_file": 1024,
                "hash": file_agg,
                "tensors": [{"name": "t", "hash": t_hash}],
            },
            {
                "file": "blk-00001.gguf",
                "kind": "layer",
                "abs_index": 1,
                "n_bytes_file": 1024,
                "hash": "11111111111111111111111111111111",
                "tensors": [],  # Whole-file scope because no per-tensor hashes
            },
        ],
    }

    with pytest.raises(ValueError, match="Inconsistent or mixed hash scopes"):
        Manifest.from_dict(manifest_dict)


def test_tensor_aggregate_verification_with_whole_file_verifier() -> None:
    """Tensor-aggregate manifest with whole-file verifier classifies scope mismatch."""
    import hashlib

    t_hash = hashlib.blake2b(b"t-data", digest_size=16).hexdigest()
    agg = hashlib.blake2b(digest_size=16)
    agg.update(bytes.fromhex(t_hash))
    tensor_agg_hash = agg.hexdigest()

    # Raw file bytes differ from tensor-aggregate hash
    raw_content = b"0" * 1000
    whole_file_hash = compute_blake2b_128(raw_content)
    assert whole_file_hash != tensor_agg_hash

    manifest_dict: dict[str, Any] = {
        "format": "gguf-layer-library/v1",
        "hash_algo": "blake2b-128",
        "hash_scope": "tensor-aggregate",
        "source": {"model_name": "test", "arch": "llama", "block_count": 1},
        "files": [
            {
                "file": "blk-00000.gguf",
                "kind": "layer",
                "abs_index": 0,
                "n_bytes_file": 1000,
                "hash": tensor_agg_hash,
            },
            {
                "file": "parts-embd.gguf",
                "kind": "embd",
                "abs_index": None,
                "n_bytes_file": 1000,
                "hash": tensor_agg_hash,
            },
            {
                "file": "parts-output.gguf",
                "kind": "output",
                "abs_index": None,
                "n_bytes_file": 1000,
                "hash": tensor_agg_hash,
            },
        ],
    }

    files = {
        "blk-00000.gguf": raw_content,
        "parts-embd.gguf": raw_content,
        "parts-output.gguf": raw_content,
    }

    # Verify with default whole-file verifier
    report = verify(files, manifest_dict)
    # Passed is True (sizes match, no missing or corrupt files)
    assert report.passed is True
    # hash_verified is False (whole-file verifier cannot verify tensor-aggregate scope)
    assert report.hash_verified is False
    assert report.scope_verified is False
    assert report.hash_scope == "tensor-aggregate"
    assert report.verified_scope is None
    assert report.summary[FileStatus.UNVERIFIED_SCOPE_MISMATCH.value] == 3
    assert report.summary[FileStatus.PASS.value] == 0
    assert report.summary[FileStatus.FAIL.value] == 0

    detail = report["blk-00000.gguf"].detail
    assert "tensor-aggregate" in detail
    assert "whole-file verifier cannot verify tensor-aggregate digests" in detail

    # If a file has size mismatch, it correctly fails
    corrupt_size_files = dict(files)
    corrupt_size_files["blk-00000.gguf"] = b"short"
    bad_report = verify(corrupt_size_files, manifest_dict)
    assert bad_report.passed is False
    assert bad_report.summary[FileStatus.SIZE_MISMATCH.value] == 1


def test_tensor_aggregate_verification_with_scope_matching() -> None:
    """When verifier scope matches manifest scope, matching hashes pass and verify."""
    manifest_dict: dict[str, Any] = {
        "format": "gguf-layer-library/v1",
        "hash_algo": "blake2b-128",
        "hash_scope": "tensor-aggregate",
        "source": {"model_name": "test", "arch": "llama", "block_count": 1},
        "files": [
            {
                "file": "blk-00000.gguf",
                "kind": "layer",
                "abs_index": 0,
                "n_bytes_file": 100,
                "hash": "11111111111111111111111111111111",
            },
            {
                "file": "parts-embd.gguf",
                "kind": "embd",
                "abs_index": None,
                "n_bytes_file": 100,
                "hash": "11111111111111111111111111111111",
            },
            {
                "file": "parts-output.gguf",
                "kind": "output",
                "abs_index": None,
                "n_bytes_file": 100,
                "hash": "11111111111111111111111111111111",
            },
        ],
    }

    # Pass pre-computed tensor digests using 3-tuple (size, hash, scope)
    files = {
        "blk-00000.gguf": (100, "11111111111111111111111111111111", "tensor-aggregate"),
        "parts-embd.gguf": (100, "11111111111111111111111111111111", "tensor-aggregate"),
        "parts-output.gguf": (100, "11111111111111111111111111111111", "tensor-aggregate"),
    }

    report = verify(files, manifest_dict, verifier_scope="tensor-aggregate")
    assert report.passed is True
    assert report.hash_verified is True
    assert report.scope_verified is True
    assert report.hash_scope == "tensor-aggregate"
    assert report.verified_scope == "tensor-aggregate"
    assert report.summary[FileStatus.PASS.value] == 3


def test_manifest_roundtrip_preserves_scope_and_tensors() -> None:
    """Manifest.to_dict preserves hash_scope and per-tensor metadata."""
    original: dict[str, Any] = {
        "format": "gguf-layer-library/v1",
        "hash_algo": "blake2b-128",
        "hash_scope": "tensor-aggregate",
        "source": {
            "model_name": "llama",
            "arch": "llama",
            "block_count": 1,
            "leading_dense_block_count": 0,
            "nextn_predict_layers": 0,
        },
        "files": [
            {
                "file": "blk-00000.gguf",
                "kind": "layer",
                "abs_index": 0,
                "n_bytes_file": 500,
                "hash": "abcdefabcdefabcdefabcdefabcdefab",
                "n_tensors": 1,
                "n_bytes_data": 400,
                "tensors": [{"name": "weight", "shape": [10, 10], "hash": "1234"}],
            }
        ],
    }

    m = Manifest.from_dict(original)
    assert m.hash_scope == "tensor-aggregate"
    assert len(m.files[0].tensors) == 1
    assert m.files[0].tensors[0]["name"] == "weight"

    roundtrip = m.to_dict()
    assert roundtrip["hash_scope"] == "tensor-aggregate"
    orig_files: list[dict[str, Any]] = original["files"]
    rt_files: list[dict[str, Any]] = roundtrip["files"]
    assert rt_files[0]["tensors"] == orig_files[0]["tensors"]
