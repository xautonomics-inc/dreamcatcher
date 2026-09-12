"""Tests for layer_distribution slicer and slice CLI."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gguf import GGUFEndian, GGUFWriter

from layer_distribution import compute_blake2b_128, slice_model, verify
from layer_distribution.slice import main as slice_cli_main
from layer_distribution.slicer import supports_tensor_endianess


def create_synthetic_gguf(
    path: Path,
    n_blocks: int = 2,
    nextn: int = 0,
    leading_dense: int = 0,
    include_other: bool = False,
    tensor_dim: int = 4,
    endianess: GGUFEndian = GGUFEndian.LITTLE,
) -> Path:
    """Create a minimal valid synthetic GGUF model for testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(path, "llama", endianess=endianess)
    w.add_uint32("llama.block_count", n_blocks)
    w.add_uint32("llama.leading_dense_block_count", leading_dense)
    w.add_uint32("llama.nextn_predict_layers", nextn)
    w.add_string("general.name", "synthetic-test-model")

    d_embd = np.arange(tensor_dim * tensor_dim, dtype=np.float32).reshape((tensor_dim, tensor_dim))
    w.add_tensor_info("token_embd.weight", d_embd.shape, d_embd.dtype, d_embd.nbytes)

    block_tensors: list[np.ndarray[tuple[int, int], np.dtype[np.float32]]] = []
    for i in range(n_blocks):
        d_blk = np.arange(tensor_dim * tensor_dim, dtype=np.float32).reshape(
            (tensor_dim, tensor_dim)
        ) * (i + 2)
        block_tensors.append(d_blk)
        w.add_tensor_info(f"blk.{i}.attn_q.weight", d_blk.shape, d_blk.dtype, d_blk.nbytes)

    d_out = (
        np.arange(tensor_dim * tensor_dim, dtype=np.float32).reshape((tensor_dim, tensor_dim)) * 10
    )
    w.add_tensor_info("output_norm.weight", d_out.shape, d_out.dtype, d_out.nbytes)

    d_other = None
    if include_other:
        d_other = np.ones((tensor_dim,), dtype=np.float32)
        w.add_tensor_info("rope_freqs.weight", d_other.shape, d_other.dtype, d_other.nbytes)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()

    def write_t(arr: np.ndarray[Any, Any]) -> None:
        if endianess == GGUFEndian.BIG:
            w.write_tensor_data(arr, tensor_endianess=endianess)
        else:
            w.write_tensor_data(arr)

    write_t(d_embd)
    for d_blk in block_tensors:
        write_t(d_blk)
    write_t(d_out)
    if d_other is not None:
        write_t(d_other)

    w.close()
    return path


def test_slice_synthetic_model_verifies_whole_file(tmp_path: Path) -> None:
    model_path = create_synthetic_gguf(tmp_path / "model.gguf", n_blocks=2)
    out_dir = tmp_path / "library"

    manifest = slice_model(str(model_path), out_dir)

    assert manifest["format"] == "gguf-layer-library/v1"
    assert manifest["hash_scope"] == "whole-file"
    assert manifest["hash_algo"] == "blake2b-128"
    assert manifest["tool"] == "layer_distribution.slice"

    files = {f["file"]: f for f in manifest["files"]}
    assert "blk-00000.gguf" in files
    assert "blk-00001.gguf" in files
    assert "parts-embd.gguf" in files
    assert "parts-output.gguf" in files

    # Verify per-file hashes match actual whole-file blake2b-128
    for fname, fentry in files.items():
        file_path = out_dir / fname
        assert file_path.is_file()
        expected_hash = compute_blake2b_128(file_path.read_bytes())
        assert fentry["hash"] == expected_hash
        assert fentry["n_bytes_file"] == file_path.stat().st_size
        assert len(fentry["tensors"]) > 0
        for trec in fentry["tensors"]:
            assert "hash" in trec
            assert len(trec["hash"]) == 32

    # Verify with L5 engine verify()
    report = verify(out_dir, manifest)
    assert report.passed is True
    assert report.hash_verified is True
    assert report.scope_verified is True
    assert report.hash_scope == "whole-file"
    assert report.verified_scope == "whole-file"
    for fname in files:
        assert report[fname].status == "pass"
        assert report[fname].passed is True
        assert report[fname].hash_verified is True


def test_corrupted_byte_refuses_verification(tmp_path: Path) -> None:
    model_path = create_synthetic_gguf(tmp_path / "model.gguf", n_blocks=2)
    out_dir = tmp_path / "library"

    manifest = slice_model(str(model_path), out_dir)
    assert verify(out_dir, manifest).passed is True

    # Corrupt one byte in blk-00000.gguf
    target_file = out_dir / "blk-00000.gguf"
    data = bytearray(target_file.read_bytes())
    data[-1] ^= 0xFF
    target_file.write_bytes(data)

    report = verify(out_dir, manifest)
    assert report.passed is False
    assert report.hash_verified is False
    assert report["blk-00000.gguf"].status == "fail"
    assert report["blk-00000.gguf"].passed is False
    assert report["blk-00000.gguf"].hash_verified is False


def test_slice_refuses_overwrite_without_force(tmp_path: Path) -> None:
    model_path = create_synthetic_gguf(tmp_path / "model.gguf", n_blocks=1)
    out_dir = tmp_path / "library"

    slice_model(str(model_path), out_dir)
    assert (out_dir / "manifest.json").exists()

    with pytest.raises(FileExistsError, match="exists and is not empty"):
        slice_model(str(model_path), out_dir)


def test_slice_force_overwrites_existing(tmp_path: Path) -> None:
    model_path = create_synthetic_gguf(tmp_path / "model.gguf", n_blocks=1)
    out_dir = tmp_path / "library"

    slice_model(str(model_path), out_dir)
    assert (out_dir / "manifest.json").exists()

    # With force=True, slicing proceeds cleanly
    manifest = slice_model(str(model_path), out_dir, force=True)
    assert (out_dir / "manifest.json").exists()
    assert verify(out_dir, manifest).passed is True


def test_slice_resume_reuses_valid_files(tmp_path: Path) -> None:
    model_path = create_synthetic_gguf(tmp_path / "model.gguf", n_blocks=3)
    out_dir = tmp_path / "library"

    # Initial full slice
    manifest1 = slice_model(str(model_path), out_dir)
    assert verify(out_dir, manifest1).passed is True

    blk0 = out_dir / "blk-00000.gguf"
    blk0_mtime = blk0.stat().st_mtime_ns

    # Remove blk-00002.gguf and corrupt blk-00001.gguf
    (out_dir / "blk-00002.gguf").unlink()
    (out_dir / "blk-00001.gguf").write_bytes(b"corrupted partial data")

    time.sleep(0.01)
    manifest2 = slice_model(str(model_path), out_dir, resume=True)

    # blk-00000.gguf was valid and complete -> reused (mtime unchanged)
    assert blk0.stat().st_mtime_ns == blk0_mtime

    # blk-00001 was corrupted and blk-00002 was missing -> re-sliced
    assert (out_dir / "blk-00001.gguf").exists()
    assert (out_dir / "blk-00002.gguf").exists()

    report = verify(out_dir, manifest2)
    assert report.passed is True
    assert report.hash_verified is True


def test_slice_rate_limiting(tmp_path: Path) -> None:
    # 256 x 256 float32 tensor = 262,144 bytes per tensor (~1 MB total model)
    model_path = create_synthetic_gguf(tmp_path / "model.gguf", n_blocks=2, tensor_dim=256)
    out_dir = tmp_path / "library"

    # Limit to 1 MiB/s
    start = time.monotonic()
    manifest = slice_model(str(model_path), out_dir, rate=1.0)
    elapsed = time.monotonic() - start

    assert elapsed >= 0.5
    assert verify(out_dir, manifest).passed is True


def test_slice_provenance_scrubbing(tmp_path: Path) -> None:
    model_path = create_synthetic_gguf(tmp_path / "nested" / "model.gguf", n_blocks=1)
    out_dir = tmp_path / "library"

    manifest = slice_model(str(model_path.resolve()), out_dir)
    # Absolute path was converted to basename
    assert manifest["source"]["glob"] == "model.gguf"
    assert "/" not in manifest["source"]["glob"]

    # Explicit source_label overrides
    out_dir2 = tmp_path / "library2"
    manifest2 = slice_model(
        str(model_path.resolve()), out_dir2, source_label="models/pinned-model-*.gguf"
    )
    assert manifest2["source"]["glob"] == "models/pinned-model-*.gguf"


def test_slice_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model_path = create_synthetic_gguf(tmp_path / "model.gguf", n_blocks=1)
    out_dir = tmp_path / "library"

    rc = slice_cli_main([str(model_path), str(out_dir)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "[lib] DONE" in captured.out
    assert (out_dir / "manifest.json").exists()

    # Test error exit code on missing model
    rc_err = slice_cli_main(["nonexistent.gguf", str(out_dir)])
    assert rc_err == 1


def test_slice_with_nextn_and_other_tensors(tmp_path: Path) -> None:
    model_path = create_synthetic_gguf(
        tmp_path / "model.gguf", n_blocks=2, nextn=1, include_other=True
    )
    out_dir = tmp_path / "library"

    manifest = slice_model(str(model_path), out_dir)
    files = {f["file"]: f for f in manifest["files"]}

    assert "blk-00000.gguf" in files
    assert "parts-nextn-00001.gguf" in files
    assert "parts-embd.gguf" in files
    assert "parts-output.gguf" in files
    assert "parts-other.gguf" in files

    report = verify(out_dir, manifest)
    assert report.passed is True
    assert report.hash_verified is True
    assert report.scope_verified is True


def test_slice_feature_detect_without_tensor_endianess_kwarg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify slicer succeeds on little-endian sources with older ik-lineage gguf-py."""
    model_path = create_synthetic_gguf(tmp_path / "model.gguf", n_blocks=2)
    out_dir = tmp_path / "library"

    orig_write = GGUFWriter.write_tensor_data

    def legacy_write(self: GGUFWriter, tensor: np.ndarray[Any, Any]) -> None:
        orig_write(self, tensor)

    monkeypatch.setattr(GGUFWriter, "write_tensor_data", legacy_write)
    assert not supports_tensor_endianess()

    manifest = slice_model(str(model_path), out_dir)
    assert len(manifest["files"]) == 4

    report = verify(out_dir, manifest)
    assert report.passed is True
    assert report.hash_verified is True
    assert report.scope_verified is True


def test_slice_big_endian_without_tensor_endianess_refuses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify slicer refuses loudly when source is big-endian but gguf-py lacks tensor_endianess."""
    model_path = create_synthetic_gguf(
        tmp_path / "model.gguf", n_blocks=1, endianess=GGUFEndian.BIG
    )
    out_dir = tmp_path / "library"

    orig_write = GGUFWriter.write_tensor_data

    def legacy_write(self: GGUFWriter, tensor: np.ndarray[Any, Any]) -> None:
        orig_write(self, tensor)

    monkeypatch.setattr(GGUFWriter, "write_tensor_data", legacy_write)
    assert not supports_tensor_endianess()

    with pytest.raises(
        ValueError, match=r"Big-endian GGUF source.*cannot be sliced.*tensor_endianess"
    ):
        slice_model(str(model_path), out_dir)
