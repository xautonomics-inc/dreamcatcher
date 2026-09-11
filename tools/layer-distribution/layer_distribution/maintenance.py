"""Bounded-I/O manifest maintenance; model bytes are never modified."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import stat
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .engine import files_for_window, verify
from .models import Manifest

# Preserve arbitrary JSON provenance beyond the typed distribution model.
JSON = dict[str, Any]


def load_manifest(root: Path) -> tuple[JSON, Manifest, bytes]:
    """Parse using the shared library and validate maintenance preconditions."""
    path = root / "manifest.json"
    if not path.exists():
        raise ValueError("No manifest: derivation refused; restore metadata or re-slice the source")
    if path.is_symlink():
        raise ValueError("Manifest symlinks are not supported")
    original = path.read_bytes()
    data = json.loads(original)
    if not isinstance(data, dict):
        raise ValueError("Manifest must be an object")
    m = Manifest.from_dict(data)
    if m.format != "gguf-layer-library/v1" or not m.files:
        raise ValueError("Expected a nonempty gguf-layer-library/v1 manifest")
    if m.hash_algo not in (None, "blake2b-128"):
        raise ValueError("Unsupported hash algorithm")
    names = [f.file for f in m.files]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate manifest filenames")
    for f in m.files:
        if Path(f.file).name != f.file or not f.file.endswith(".gguf"):
            raise ValueError(f"Unsafe filename: {f.file}")
        if f.n_bytes_file <= 0:
            raise ValueError(f"Invalid size: {f.file}")
        if f.hash and (len(f.hash) != 32 or any(c not in "0123456789abcdef" for c in f.hash)):
            raise ValueError(f"Invalid blake2b-128 hash: {f.file}")
    required = files_for_window(m, 0, m.source.block_count)
    if not required <= set(names):
        raise ValueError(f"Incomplete layer coverage: {sorted(required - set(names))}")
    return data, m, original


def legacy_aggregate_files(data: JSON) -> list[str]:
    """Recognize the old slicer's digest-of-tensor-digests convention, without I/O."""
    found: list[str] = []
    for entry in data["files"]:
        tensors = entry.get("tensors", [])
        if not entry.get("hash") or not tensors:
            continue
        digests = [t.get("hash") for t in tensors]
        if not all(
            isinstance(h, str) and len(h) == 32 and all(c in "0123456789abcdef" for c in h)
            for h in digests
        ):
            continue
        aggregate = hashlib.blake2b(digest_size=16)
        for digest in digests:
            aggregate.update(bytes.fromhex(digest))
        if aggregate.hexdigest() == entry["hash"]:
            found.append(str(entry["file"]))
    return found


def fingerprint(path: Path) -> list[int]:
    """Reject links and capture file identity, size and mutation timestamps."""
    s = path.lstat()
    if not stat.S_ISREG(s.st_mode):
        raise ValueError(f"Not a regular file: {path.name}")
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def atomic_json(path: Path, data: JSON, *, mode: int = 0o600) -> None:
    """Write and fsync a complete sibling before atomic replacement."""
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            os.fchmod(stream.fileno(), mode)
            json.dump(data, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextmanager
def state_lock(state: Path, filename: str = "lock") -> Iterator[None]:
    """Serialize callers sharing a checkpoint directory."""
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (state / filename).open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def hash_file(path: Path, rate: float) -> tuple[int, str]:
    """Stream with a one-MiB burst bound and monotonic MiB/s pacing."""
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("Rate must be finite and positive")
    before = fingerprint(path)
    h = hashlib.blake2b(digest_size=16)
    size = 0
    started = time.monotonic()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        s = os.fstat(stream.fileno())
        if [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns] != before:
            raise ValueError(f"File changed before read: {path.name}")
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
            size += len(chunk)
            delay = size / (rate * 1024 * 1024) - (time.monotonic() - started)
            if delay > 0:
                time.sleep(delay)
    if fingerprint(path) != before or size != before[2]:
        raise ValueError(f"File changed during read: {path.name}")
    return size, h.hexdigest()


def inventory(root: Path) -> JSON:
    """Inspect membership and sizes without reading weight data."""
    present = {p.name for p in root.glob("*.gguf")}
    report: JSON = {"library": root.name, "observed_at": time.time(), "present": len(present)}
    try:
        data, m, _ = load_manifest(root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {**report, "status": "refused", "reason": str(exc), "hash_verified": False}
    legacy = legacy_aggregate_files(data)
    expected = {f.file for f in m.files}
    mismatches: list[str] = []
    unsafe: list[str] = []
    for f in m.files:
        if f.file in present:
            try:
                if fingerprint(root / f.file)[2] != f.n_bytes_file:
                    mismatches.append(f.file)
            except (OSError, ValueError):
                unsafe.append(f.file)
    return {
        **report,
        "status": (
            "inventory-failed"
            if expected != present or mismatches or unsafe
            else "legacy-hash-scope"
            if legacy
            else "inventory-only"
        ),
        "manifest_entries": len(expected),
        "existing_hashes": sum(f.hash is not None for f in m.files),
        "legacy_tensor_aggregate_files": legacy,
        "hash_scope": "tensor-aggregate" if legacy else data.get("hash_scope", "unmarked"),
        "missing": sorted(expected - present),
        "unexpected": sorted(present - expected),
        "size_mismatches": mismatches,
        "unsafe": unsafe,
        "hash_verified": False,
    }


def prepare(root: Path, state: Path, rate: float, *, fresh: bool = False) -> JSON:
    """Resume hashing into external state; never replace known mismatching hashes."""
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("Rate must be finite and positive")
    root, state = root.resolve(), state.resolve()
    if state == root or root in state.parents:
        raise ValueError("State directory must be outside the library")
    with state_lock(state):
        data, m, original = load_manifest(root)
        inv = inventory(root)
        if inv.get("legacy_tensor_aggregate_files"):
            raise ValueError(
                "Legacy tensor-aggregate hashes are not whole-file hashes; "
                "refusing rebaseline. Verify tensors against their original hashes "
                "before a separately reviewed migration."
            )
        if any(inv.get(k) for k in ("missing", "unexpected", "size_mismatches", "unsafe")):
            raise ValueError(f"Inventory failed: {json.dumps(inv)}")
        identity = hashlib.sha256(original).hexdigest()
        checkpoint = state / "checkpoint.json"
        saved: JSON = {"library": str(root), "manifest_sha256": identity, "files": {}}
        if checkpoint.exists() and not fresh:
            old = json.loads(checkpoint.read_bytes())
            if old.get("library") == str(root) and old.get("manifest_sha256") == identity:
                saved = old
        actual: dict[str, tuple[int, str]] = {}
        stamps: dict[str, list[int]] = {}
        for f in m.files:
            path = root / f.file
            stamp = fingerprint(path)
            cached = saved["files"].get(f.file)
            if cached and cached.get("fingerprint") == stamp:
                size, digest = cached["size"], cached["hash"]
            else:
                size, digest = hash_file(path, rate)
                if fingerprint(path) != stamp:
                    raise ValueError(f"File changed: {f.file}")
                saved["files"][f.file] = {"fingerprint": stamp, "size": size, "hash": digest}
                atomic_json(checkpoint, saved)
            actual[f.file] = (size, digest)
            stamps[f.file] = stamp
            print(f"Hashed {len(actual)}/{len(m.files)}: {f.file}", file=sys.stderr, flush=True)
        before = verify(actual, m)
        if not before.passed:
            atomic_json(state / "verification.json", before.to_dict())
            raise ValueError("Existing size/hash mismatch; original manifest retained")
        candidate = copy.deepcopy(data)
        candidate["hash_algo"] = "blake2b-128"
        candidate["hash_scope"] = "whole-file"
        for entry in candidate["files"]:
            entry["hash"] = actual[entry["file"]][1]
        after = verify(actual, Manifest.from_dict(candidate))
        if not after.hash_verified:
            raise ValueError("Candidate failed verification")
        if (root / "manifest.json").read_bytes() != original:
            raise ValueError("Manifest changed during preparation")
        for name, stamp in stamps.items():
            if fingerprint(root / name) != stamp:
                raise ValueError(f"File changed during preparation: {name}")
        atomic_json(state / "candidate.json", candidate)
        report = {
            "library": str(root),
            "observed_at": time.time(),
            "manifest_sha256": identity,
            "fingerprints": stamps,
            "candidate_sha256": hashlib.sha256((state / "candidate.json").read_bytes()).hexdigest(),
            "before": before.to_dict(),
            "after": after.to_dict(),
            "baseline_only": any(f.hash is None for f in m.files),
        }
        atomic_json(state / "verification.json", report)
        return report


def install(root: Path, state: Path, *, confirmed_idle: bool = False) -> None:
    """Install after the caller excludes consumers and concurrent writers across hosts."""
    if not confirmed_idle:
        raise ValueError("Installation requires --confirmed-idle after checking all consumers")
    root, state = root.resolve(), state.resolve()
    if state == root or root in state.parents:
        raise ValueError("State directory must be outside the library")
    with state_lock(state), state_lock(root, ".manifest-maintenance.lock"):
        report = json.loads((state / "verification.json").read_bytes())
        candidate_bytes = (state / "candidate.json").read_bytes()
        if report["library"] != str(root):
            raise ValueError("Candidate belongs to a different library")
        if hashlib.sha256(candidate_bytes).hexdigest() != report["candidate_sha256"]:
            raise ValueError("Candidate changed since verification")
        _, m, original = load_manifest(root)
        if hashlib.sha256(original).hexdigest() != report["manifest_sha256"]:
            if original == candidate_bytes:
                return
            raise ValueError("Original manifest changed since verification")
        if set(report["fingerprints"]) != {f.file for f in m.files}:
            raise ValueError("Incomplete prepared file set")
        if {p.name for p in root.glob("*.gguf")} != set(report["fingerprints"]):
            raise ValueError("Library file set changed since verification")
        for name, stamp in report["fingerprints"].items():
            if fingerprint(root / name) != stamp:
                raise ValueError(f"File changed since verification: {name}")
        candidate = json.loads(candidate_bytes)
        if not report["after"]["hash_verified"]:
            raise ValueError("Candidate was not verified")
        backup = root / ("manifest.json.before-" + report["manifest_sha256"][:16])
        if backup.exists():
            if backup.read_bytes() != original:
                raise ValueError("Backup conflict")
        else:
            with backup.open("xb") as stream:
                stream.write(original)
                stream.flush()
                os.fsync(stream.fileno())
        atomic_json(
            root / "manifest.json",
            candidate,
            mode=stat.S_IMODE((root / "manifest.json").stat().st_mode),
        )


def main() -> int:
    """CLI for inspection, preparation and explicit installation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("inventory", "prepare", "manifest-hash-upgrade", "install", "manifest-derive"),
    )
    parser.add_argument("library", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--mib-per-second", type=float, default=64)
    parser.add_argument(
        "--fresh", action="store_true", help="Ignore cached hashes for a fresh read"
    )
    parser.add_argument("--confirmed-idle", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "manifest-derive":
            raise ValueError("Derivation refused: restore a manifest or re-slice the source")
        if args.command == "inventory":
            result = inventory(args.library)
            print(json.dumps(result, indent=2))
            return 0 if result["status"] == "inventory-only" else 2
        if args.state is None:
            parser.error("--state is required")
        if args.command in ("prepare", "manifest-hash-upgrade"):
            os.nice(10)
            result = prepare(args.library, args.state, args.mib_per_second, fresh=args.fresh)
            print(json.dumps(result, indent=2))
        else:
            install(args.library, args.state, confirmed_idle=args.confirmed_idle)
            print(json.dumps({"installed": True}))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
