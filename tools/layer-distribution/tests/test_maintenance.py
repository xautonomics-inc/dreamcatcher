"""Maintenance must never bless corruption or replace a partially hashed manifest."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from layer_distribution import maintenance as mm
from layer_distribution import verify


def library(tmp_path: Path) -> Path:
    root = tmp_path / "library"
    root.mkdir()
    files = []
    for name, kind in [
        ("blk-00000.gguf", "layer"),
        ("parts-embd.gguf", "embd"),
        ("parts-output.gguf", "output"),
    ]:
        payload = name.encode() * 20
        (root / name).write_bytes(payload)
        files.append(
            {
                "file": name,
                "kind": kind,
                "abs_index": 0 if kind == "layer" else None,
                "n_bytes_file": len(payload),
                "tensors": [{"custom": "preserved"}],
            }
        )
    data = {
        "format": "gguf-layer-library/v1",
        "hash_algo": None,
        "source": {"block_count": 1, "glob": "relative/source.gguf"},
        "custom_provenance": {"keep": True},
        "files": files,
    }
    (root / "manifest.json").write_text(json.dumps(data))
    return root


def test_upgrade_preserves_metadata_and_backup(tmp_path: Path) -> None:
    root = library(tmp_path)
    original = (root / "manifest.json").read_bytes()
    state = tmp_path / "state"
    report = mm.prepare(root, state, 100)
    assert report["before"]["hash_verified"] is False
    assert report["after"]["hash_verified"] is True
    assert (root / "manifest.json").read_bytes() == original
    with pytest.raises(ValueError, match="confirmed-idle"):
        mm.install(root, state)
    mm.install(root, state, confirmed_idle=True)
    data = json.loads((root / "manifest.json").read_bytes())
    old = json.loads(original)
    assert data["source"] == old["source"]
    assert data["custom_provenance"] == old["custom_provenance"]
    assert data["files"][0]["tensors"] == old["files"][0]["tensors"]
    assert verify(root, data).hash_verified
    assert next(root.glob("manifest.json.before-*")).read_bytes() == original
    stamp = (root / "manifest.json").stat().st_mtime_ns
    mm.install(root, state, confirmed_idle=True)
    assert (root / "manifest.json").stat().st_mtime_ns == stamp


def test_resume_after_interruption(tmp_path: Path) -> None:
    root = library(tmp_path)
    state = tmp_path / "state"
    real = mm.hash_file
    calls: list[str] = []

    def interrupted(path: Path, rate: float) -> tuple[int, str]:
        calls.append(path.name)
        if len(calls) == 2:
            raise OSError("interrupted")
        return real(path, rate)

    with (
        patch.object(mm, "hash_file", side_effect=interrupted),
        pytest.raises(OSError, match="interrupted"),
    ):
        mm.prepare(root, state, 100)
    assert not (state / "candidate.json").exists()
    with patch.object(mm, "hash_file", wraps=real) as hashed:
        mm.prepare(root, state, 100)
        assert hashed.call_count == 2
    with patch.object(mm, "hash_file", wraps=real) as hashed:
        mm.prepare(root, state, 100, fresh=True)
        assert hashed.call_count == 3


def test_known_corruption_is_not_rebaselined(tmp_path: Path) -> None:
    root = library(tmp_path)
    path = root / "manifest.json"
    data = json.loads(path.read_bytes())
    data["hash_algo"] = "blake2b-128"
    for f in data["files"]:
        f["hash"] = hashlib.blake2b((root / f["file"]).read_bytes(), digest_size=16).hexdigest()
    path.write_text(json.dumps(data))
    original = path.read_bytes()
    damaged = root / "blk-00000.gguf"
    damaged.write_bytes(b"x" * damaged.stat().st_size)
    with pytest.raises(ValueError, match="mismatch"):
        mm.prepare(root, tmp_path / "state", 100)
    assert path.read_bytes() == original
    assert not (tmp_path / "state" / "candidate.json").exists()


@pytest.mark.parametrize("change", ["file", "manifest", "candidate", "extra"])
def test_install_rejects_changes(tmp_path: Path, change: str) -> None:
    root = library(tmp_path)
    state = tmp_path / "state"
    mm.prepare(root, state, 100)
    if change == "file":
        p = root / "blk-00000.gguf"
        p.write_bytes(b"z" * p.stat().st_size)
    elif change == "manifest":
        p = root / "manifest.json"
        p.write_bytes(p.read_bytes() + b" ")
    elif change == "candidate":
        p = state / "candidate.json"
        p.write_bytes(p.read_bytes() + b" ")
    else:
        (root / "extra.gguf").write_bytes(b"x")
    with pytest.raises(ValueError, match="changed"):
        mm.install(root, state, confirmed_idle=True)
    assert not list(root.glob("manifest.json.before-*"))


@pytest.mark.parametrize("change", ["missing", "size", "symlink", "duplicate", "traversal"])
def test_prepare_refuses_invalid_libraries(tmp_path: Path, change: str) -> None:
    root = library(tmp_path)
    p = root / "blk-00000.gguf"
    if change == "missing":
        p.unlink()
    elif change == "size":
        p.write_bytes(b"x")
    elif change == "symlink":
        p.unlink()
        p.symlink_to(root / "parts-embd.gguf")
    else:
        m = root / "manifest.json"
        data = json.loads(m.read_bytes())
        if change == "duplicate":
            data["files"].append(data["files"][0])
        else:
            data["files"][0]["file"] = "../outside.gguf"
        m.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        mm.prepare(root, tmp_path / "state", 100)


def test_no_manifest_and_nested_state_refused(tmp_path: Path) -> None:
    root = library(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        mm.prepare(root, root / "state", 100)
    (root / "manifest.json").unlink()
    assert mm.inventory(root)["status"] == "refused"
    with pytest.raises(ValueError, match="No manifest"):
        mm.prepare(root, tmp_path / "state", 100)


def test_rate_and_mutation_detection(tmp_path: Path) -> None:
    path = tmp_path / "weights"
    path.write_bytes(b"x" * 1024 * 1024)
    for rate in [0.0, -1.0, float("nan"), float("inf")]:
        with pytest.raises(ValueError, match="Rate"):
            mm.hash_file(path, rate)
    with (
        patch.object(time, "monotonic", return_value=0),
        patch.object(time, "sleep") as sleep,
    ):
        size, _ = mm.hash_file(path, 2)
        assert size == 1024 * 1024
        sleep.assert_called_once_with(0.5)
    before = mm.fingerprint(path)
    with (
        patch.object(mm, "fingerprint", side_effect=[before, [0, 0, 0, 0, 0]]),
        pytest.raises(ValueError, match="changed during"),
    ):
        mm.hash_file(path, 100)


def test_resume_invalidates_changed_file_and_manifest(tmp_path: Path) -> None:
    root = library(tmp_path)
    state = tmp_path / "state"
    mm.prepare(root, state, 100)
    path = root / "blk-00000.gguf"
    path.write_bytes(b"a" * path.stat().st_size)
    with patch.object(mm, "hash_file", wraps=mm.hash_file) as hashed:
        mm.prepare(root, state, 100)
        assert hashed.call_count == 1
    manifest = root / "manifest.json"
    manifest.write_bytes(manifest.read_bytes() + b" ")
    with patch.object(mm, "hash_file", wraps=mm.hash_file) as hashed:
        mm.prepare(root, state, 100)
        assert hashed.call_count == 3


def test_atomic_failure_keeps_original_and_permissions(tmp_path: Path) -> None:
    root = library(tmp_path)
    state = tmp_path / "state"
    manifest = root / "manifest.json"
    manifest.chmod(0o640)
    original = manifest.read_bytes()
    mm.prepare(root, state, 100)
    with (
        patch("os.replace", side_effect=OSError("disk error")),
        pytest.raises(OSError, match="disk error"),
    ):
        mm.install(root, state, confirmed_idle=True)
    assert manifest.read_bytes() == original
    mm.install(root, state, confirmed_idle=True)
    assert manifest.stat().st_mode & 0o777 == 0o640
    assert verify(root, json.loads(manifest.read_bytes())).hash_verified


def test_locks_exclude_concurrent_writers(tmp_path: Path) -> None:
    root = library(tmp_path)
    state = tmp_path / "state"
    mm.prepare(root, state, 100)
    with mm.state_lock(state), pytest.raises(BlockingIOError):
        mm.prepare(root, state, 100)
    with mm.state_lock(root, ".manifest-maintenance.lock"), pytest.raises(BlockingIOError):
        mm.install(root, state, confirmed_idle=True)


def test_cli_exit_statuses_and_explicit_install(tmp_path: Path) -> None:
    root = library(tmp_path)
    state = tmp_path / "state"
    with patch("sys.argv", ["maintenance", "inventory", str(root)]):
        assert mm.main() == 0
    with patch("sys.argv", ["maintenance", "manifest-derive", str(root)]):
        assert mm.main() == 2
    with (
        patch(
            "sys.argv", ["maintenance", "manifest-hash-upgrade", str(root), "--state", str(state)]
        ),
        patch("os.nice") as nice,
    ):
        assert mm.main() == 0
        nice.assert_called_once_with(10)
    with patch("sys.argv", ["maintenance", "install", str(root), "--state", str(state)]):
        assert mm.main() == 2
    with patch(
        "sys.argv", ["maintenance", "install", str(root), "--state", str(state), "--confirmed-idle"]
    ):
        assert mm.main() == 0
    (root / "parts-output.gguf").unlink()
    with patch("sys.argv", ["maintenance", "inventory", str(root)]):
        assert mm.main() == 2


def test_legacy_tensor_aggregate_is_not_a_whole_file_hash(tmp_path: Path) -> None:
    root = library(tmp_path)
    path = root / "manifest.json"
    data = json.loads(path.read_bytes())
    data["hash_algo"] = "blake2b-128"
    for entry in data["files"]:
        digest = hashlib.blake2b((root / entry["file"]).read_bytes(), digest_size=16).digest()
        entry["tensors"] = [{"name": "weight", "hash": digest.hex()}]
        entry["hash"] = hashlib.blake2b(digest, digest_size=16).hexdigest()
    path.write_text(json.dumps(data))
    original = path.read_bytes()
    report = mm.inventory(root)
    assert report["status"] == "legacy-hash-scope"
    assert len(report["legacy_tensor_aggregate_files"]) == 3
    assert report["missing"] == report["size_mismatches"] == []
    assert not verify(root, data).hash_verified
    with patch.object(mm, "hash_file") as hashed:
        with pytest.raises(ValueError, match="Legacy tensor-aggregate"):
            mm.prepare(root, tmp_path / "state", 100)
        hashed.assert_not_called()
    assert path.read_bytes() == original
