"""Pure, deterministic layer-distribution engine for gguf-layer-library/v1.

Implements the four core lifecycle operations:
1. files_for_window — derives stage file sets ensuring embd/output and MTP coverage
2. verify           — strict blake2b-128 and size verification against manifest
3. precheck         — space feasibility analysis with explicit arithmetic attached
4. rebalance        — delta calculation reporting files to add and removable candidates
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO

from .models import (
    FileStatus,
    FileVerifyResult,
    HashScope,
    Manifest,
    NodeDelta,
    PrecheckResult,
    RebalancePlan,
    VerifyReport,
)


def compute_blake2b_128(data: bytes) -> str:
    """Compute blake2b-128 hex digest (16 bytes / 32 hex chars) over raw bytes."""
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def compute_blake2b_128_stream(stream: BinaryIO, chunk_size: int = 65536) -> tuple[int, str]:
    """Compute total byte size and blake2b-128 hex digest over a readable binary stream."""
    hasher = hashlib.blake2b(digest_size=16)
    total_bytes = 0
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        total_bytes += len(chunk)
        hasher.update(chunk)
    return total_bytes, hasher.hexdigest()


def _normalize_manifest(manifest: dict[str, Any] | Manifest | Path | str) -> Manifest:
    """Ensure manifest input is represented as a typed Manifest model."""
    if isinstance(manifest, Manifest):
        return manifest
    if isinstance(manifest, (Path, str)):
        with open(manifest, encoding="utf-8") as f:
            data = json.load(f)
        return Manifest.from_dict(data)
    return Manifest.from_dict(manifest)


def files_for_window(manifest: dict[str, Any] | Manifest | Path | str, a: int, b: int) -> set[str]:
    """Derive the exact file set a node requires for stage window [a, b).

    The returned set contains:
    - blk-A..B-1 for standard transformer layers covered by [a, b)
    - parts-nextn-* for any MTP layers covered by [a, b)
    - parts-embd.gguf on EVERY stage (loader requirement)
    - parts-output.gguf on EVERY stage (loader requirement)
    - parts-other.gguf if present in manifest (model-level non-layer tensors)

    Args:
        manifest: Dictionary or Manifest instance defining the layer library.
        a: Start layer index (inclusive, 0-indexed).
        b: End layer index (exclusive, 0-indexed).

    Returns:
        set of required filenames.

    Raises:
        ValueError: If window bounds are negative, inverted, or exceed model block count.
    """
    m = _normalize_manifest(manifest)
    total_blocks = m.source.block_count
    nextn_count = m.source.nextn_predict_layers
    nextn_lo = total_blocks - nextn_count

    if a < 0 or b > total_blocks or a >= b:
        raise ValueError(
            f"Invalid layer window [{a}, {b}): must satisfy 0 <= a < b <= {total_blocks}"
        )

    layer_map: dict[int, str] = {}
    nextn_map: dict[int, str] = {}
    embd_file: str | None = None
    output_file: str | None = None
    other_files: list[str] = []

    for mf in m.files:
        if mf.kind == "layer" and mf.abs_index is not None:
            layer_map[mf.abs_index] = mf.file
        elif mf.kind == "nextn" and mf.abs_index is not None:
            nextn_map[mf.abs_index] = mf.file
        elif mf.kind == "embd":
            embd_file = mf.file
        elif mf.kind == "output":
            output_file = mf.file
        elif mf.kind == "other":
            other_files.append(mf.file)

    result_files: set[str] = set()

    if embd_file:
        result_files.add(embd_file)
    else:
        result_files.add("parts-embd.gguf")

    if output_file:
        result_files.add(output_file)
    else:
        result_files.add("parts-output.gguf")

    result_files.update(other_files)

    for i in range(a, b):
        if i >= nextn_lo:
            if i in nextn_map:
                result_files.add(nextn_map[i])
            else:
                result_files.add(f"parts-nextn-{i:05d}.gguf")
        else:
            if i in layer_map:
                result_files.add(layer_map[i])
            else:
                result_files.add(f"blk-{i:05d}.gguf")

    return result_files


def verify(
    files: Mapping[str, Path | bytes | tuple[int, str] | tuple[int, str, str]]
    | Path
    | str
    | Sequence[Path | str],
    manifest: dict[str, Any] | Manifest | Path | str,
    expected_files: Iterable[str] | None = None,
    verifier_scope: str = HashScope.WHOLE_FILE.value,
) -> VerifyReport:
    """Verify integrity of files against manifest using blake2b-128 and size checks.

    Supports scope-aware verification (whole-file vs tensor-aggregate). If the manifest
    specifies a hash_scope that differs from the verifier_scope (e.g. legacy slicer
    tensor-aggregate hashes checked with whole-file stream), files with matching sizes are
    classified as UNVERIFIED_SCOPE_MISMATCH (passed=True, hash_verified=False) rather than
    falsely failing as corrupt.
    """
    m = _normalize_manifest(manifest)
    manifest_by_name = {mf.file: mf for mf in m.files}

    targets: list[str]
    if expected_files is not None:
        targets = list(expected_files)
    elif isinstance(files, Mapping):
        targets = [m.files[i].file for i in range(len(m.files))]
    else:
        targets = [mf.file for mf in m.files]

    results: dict[str, FileVerifyResult] = {}
    counts: dict[str, int] = {
        FileStatus.PASS.value: 0,
        FileStatus.FAIL.value: 0,
        FileStatus.MISSING.value: 0,
        FileStatus.SIZE_MISMATCH.value: 0,
        FileStatus.UNVERIFIED_NO_HASH.value: 0,
        FileStatus.UNVERIFIED_SCOPE_MISMATCH.value: 0,
    }

    dir_path: Path | None = None
    file_map: Mapping[str, Any] | None = None

    if isinstance(files, (Path, str)):
        dir_path = Path(files)
    elif isinstance(files, Mapping):
        file_map = files
    elif isinstance(files, Sequence):
        temp_map: dict[str, Path] = {}
        for item in files:
            p = Path(item)
            temp_map[p.name] = p
        file_map = temp_map

    for fname in targets:
        mf = manifest_by_name.get(fname)
        expected_size = mf.n_bytes_file if mf else 0
        expected_hash = mf.hash if mf else None
        expected_scope = m.hash_scope

        actual_size: int | None = None
        actual_hash: str | None = None
        actual_scope: str = verifier_scope
        status: str

        if file_map is not None:
            if fname not in file_map:
                status = FileStatus.MISSING.value
                counts[status] += 1
                results[fname] = FileVerifyResult(
                    file=fname,
                    status=status,
                    expected_size=expected_size,
                    expected_hash=expected_hash,
                    expected_scope=expected_scope,
                    actual_scope=actual_scope,
                    detail=f"File '{fname}' missing from file listing",
                )
                continue

            entry = file_map[fname]
            if isinstance(entry, bytes):
                actual_size = len(entry)
                actual_hash = compute_blake2b_128(entry)
                actual_scope = verifier_scope
            elif isinstance(entry, tuple):
                if len(entry) == 2:
                    actual_size, actual_hash = entry[0], str(entry[1])
                    actual_scope = verifier_scope
                elif len(entry) == 3:
                    actual_size, actual_hash, actual_scope = entry[0], str(entry[1]), str(entry[2])
                else:
                    status = FileStatus.FAIL.value
                    counts[status] += 1
                    results[fname] = FileVerifyResult(
                        file=fname,
                        status=status,
                        expected_size=expected_size,
                        expected_hash=expected_hash,
                        expected_scope=expected_scope,
                        actual_scope=actual_scope,
                        detail=f"Invalid tuple length for file entry: {len(entry)}",
                    )
                    continue
            elif isinstance(entry, (Path, str)):
                p = Path(entry)
                if not p.is_file():
                    status = FileStatus.MISSING.value
                    counts[status] += 1
                    results[fname] = FileVerifyResult(
                        file=fname,
                        status=status,
                        expected_size=expected_size,
                        expected_hash=expected_hash,
                        expected_scope=expected_scope,
                        actual_scope=actual_scope,
                        detail=f"File '{p}' does not exist on disk",
                    )
                    continue
                with open(p, "rb") as stream:
                    actual_size, actual_hash = compute_blake2b_128_stream(stream)
                actual_scope = verifier_scope
            else:
                status = FileStatus.FAIL.value
                counts[status] += 1
                results[fname] = FileVerifyResult(
                    file=fname,
                    status=status,
                    expected_size=expected_size,
                    expected_hash=expected_hash,
                    expected_scope=expected_scope,
                    actual_scope=actual_scope,
                    detail=f"Unsupported file entry type: {type(entry)}",
                )
                continue

        elif dir_path is not None:
            p = dir_path / fname
            if not p.is_file():
                status = FileStatus.MISSING.value
                counts[status] += 1
                results[fname] = FileVerifyResult(
                    file=fname,
                    status=status,
                    expected_size=expected_size,
                    expected_hash=expected_hash,
                    expected_scope=expected_scope,
                    actual_scope=actual_scope,
                    detail=f"File '{fname}' missing from directory '{dir_path}'",
                )
                continue
            with open(p, "rb") as stream:
                actual_size, actual_hash = compute_blake2b_128_stream(stream)
            actual_scope = verifier_scope

        else:
            status = FileStatus.MISSING.value
            counts[status] += 1
            results[fname] = FileVerifyResult(
                file=fname,
                status=status,
                expected_size=expected_size,
                expected_hash=expected_hash,
                expected_scope=expected_scope,
                actual_scope=actual_scope,
                detail="No files provided",
            )
            continue

        if actual_size != expected_size:
            status = FileStatus.SIZE_MISMATCH.value
            detail = f"Size mismatch: expected {expected_size} bytes, got {actual_size} bytes"
        elif expected_hash is None:
            status = FileStatus.UNVERIFIED_NO_HASH.value
            detail = (
                f"Size matches ({actual_size} bytes), but manifest has no hash for verification"
            )
        else:
            # We have an expected hash. Check scope compatibility.
            if expected_scope is not None and expected_scope != actual_scope:
                # Scope mismatch: classify rather than reporting false corruption!
                status = FileStatus.UNVERIFIED_SCOPE_MISMATCH.value
                detail = (
                    f"Size matches ({actual_size} bytes), but manifest hash_scope is "
                    f"'{expected_scope}'; {actual_scope} verifier cannot verify "
                    f"{expected_scope} digests without tensor parser"
                )
            elif actual_hash != expected_hash:
                status = FileStatus.FAIL.value
                detail = (
                    f"Hash mismatch ({actual_scope}): expected {expected_hash}, got {actual_hash}"
                )
            else:
                status = FileStatus.PASS.value
                algo = m.hash_algo or "blake2b-128"
                detail = f"Verification passed (size and {algo} {actual_scope} match)"

        counts[status] += 1
        results[fname] = FileVerifyResult(
            file=fname,
            status=status,
            expected_size=expected_size,
            actual_size=actual_size,
            expected_hash=expected_hash,
            actual_hash=actual_hash,
            expected_scope=expected_scope,
            actual_scope=actual_scope,
            detail=detail,
        )

    has_errors = (
        counts[FileStatus.MISSING.value] > 0
        or counts[FileStatus.SIZE_MISMATCH.value] > 0
        or counts[FileStatus.FAIL.value] > 0
    )
    all_passed = not has_errors and len(results) > 0
    all_hash_verified = (
        all_passed
        and counts[FileStatus.UNVERIFIED_NO_HASH.value] == 0
        and counts[FileStatus.UNVERIFIED_SCOPE_MISMATCH.value] == 0
        and counts[FileStatus.PASS.value] > 0
    )
    verified_scope = verifier_scope if all_hash_verified else None

    return VerifyReport(
        passed=all_passed,
        hash_verified=all_hash_verified,
        files=results,
        summary=counts,
        hash_scope=m.hash_scope,
        verified_scope=verified_scope,
    )


def precheck(
    files: Iterable[str],
    manifest: dict[str, Any] | Manifest | Path | str,
    free_bytes: int,
) -> PrecheckResult:
    """Check destination disk space feasibility with explicit arithmetic."""
    if free_bytes < 0:
        raise ValueError(f"free_bytes cannot be negative (got {free_bytes})")

    m = _normalize_manifest(manifest)
    manifest_by_name = {mf.file: mf for mf in m.files}

    per_file: dict[str, int] = {}
    required_bytes = 0
    for f in sorted(files):
        mf = manifest_by_name.get(f)
        if mf is None:
            raise ValueError(f"File '{f}' not declared in manifest")
        size = mf.n_bytes_file
        per_file[f] = size
        required_bytes += size

    feasible = free_bytes >= required_bytes
    if feasible:
        surplus_bytes = free_bytes - required_bytes
        deficit_bytes = 0
        reason = (
            f"Feasible: required {required_bytes} bytes, free {free_bytes} bytes "
            f"(surplus: {surplus_bytes} bytes)"
        )
    else:
        surplus_bytes = 0
        deficit_bytes = required_bytes - free_bytes
        reason = (
            f"Infeasible: required {required_bytes} bytes, free {free_bytes} bytes "
            f"(shortfall: {deficit_bytes} bytes)"
        )

    return PrecheckResult(
        feasible=feasible,
        required_bytes=required_bytes,
        free_bytes=free_bytes,
        deficit_bytes=deficit_bytes,
        surplus_bytes=surplus_bytes,
        reason=reason,
        per_file_bytes=per_file,
    )


def rebalance(
    old_windows: Mapping[str, tuple[int, int] | None],
    new_windows: Mapping[str, tuple[int, int] | None],
    manifest: dict[str, Any] | Manifest | Path | str,
) -> RebalancePlan:
    """Compute per-node delta across old and new window assignments."""
    m = _normalize_manifest(manifest)
    file_sizes = {mf.file: mf.n_bytes_file for mf in m.files}

    all_nodes = sorted(set(old_windows.keys()) | set(new_windows.keys()))
    node_deltas: dict[str, NodeDelta] = {}
    total_bytes_add = 0
    total_bytes_removable = 0

    for node in all_nodes:
        old_w = old_windows.get(node)
        new_w = new_windows.get(node)

        old_files: set[str] = set()
        if old_w is not None:
            old_files = files_for_window(m, old_w[0], old_w[1])

        new_files: set[str] = set()
        if new_w is not None:
            new_files = files_for_window(m, new_w[0], new_w[1])

        files_to_add = sorted(new_files - old_files)
        files_removable = sorted(old_files - new_files)

        bytes_to_add = sum(file_sizes.get(f, 0) for f in files_to_add)
        bytes_removable = sum(file_sizes.get(f, 0) for f in files_removable)

        total_bytes_add += bytes_to_add
        total_bytes_removable += bytes_removable

        node_deltas[node] = NodeDelta(
            node=node,
            old_window=old_w,
            new_window=new_w,
            files_to_add=files_to_add,
            files_removable=files_removable,
            bytes_to_add=bytes_to_add,
            bytes_removable=bytes_removable,
        )

    return RebalancePlan(
        nodes=node_deltas,
        total_bytes_to_add=total_bytes_add,
        total_bytes_removable=total_bytes_removable,
    )


def enrich_manifest_hashes(
    manifest: dict[str, Any] | Manifest | Path | str,
    files: Mapping[str, Path | bytes | tuple[int, str]] | Path | str | Sequence[Path | str],
    hash_scope: str = HashScope.WHOLE_FILE.value,
) -> Manifest:
    """Upgrade a hashless manifest by computing blake2b-128 checksums from source files.

    Returns a new Manifest instance populated with computed hashes, hash_algo set
    to 'blake2b-128', and hash_scope set to 'whole-file'. Useful for upgrading
    legacy or unhashed layer libraries once.
    """
    m = _normalize_manifest(manifest)
    enriched_files = []

    dir_path: Path | None = None
    file_map: Mapping[str, Any] | None = None

    if isinstance(files, (Path, str)):
        dir_path = Path(files)
    elif isinstance(files, Mapping):
        file_map = files
    elif isinstance(files, Sequence):
        temp_map: dict[str, Path] = {}
        for item in files:
            p = Path(item)
            temp_map[p.name] = p
        file_map = temp_map

    from .models import ManifestFile

    for mf in m.files:
        h = mf.hash
        size = mf.n_bytes_file

        if not h:
            if file_map is not None and mf.file in file_map:
                entry = file_map[mf.file]
                if isinstance(entry, bytes):
                    h = compute_blake2b_128(entry)
                    size = len(entry)
                elif isinstance(entry, tuple) and len(entry) == 2:
                    size, h = entry[0], str(entry[1])
                elif isinstance(entry, (Path, str)):
                    p = Path(entry)
                    if p.is_file():
                        with open(p, "rb") as stream:
                            size, h = compute_blake2b_128_stream(stream)
            elif dir_path is not None:
                p = dir_path / mf.file
                if p.is_file():
                    with open(p, "rb") as stream:
                        size, h = compute_blake2b_128_stream(stream)

        enriched_files.append(
            ManifestFile(
                file=mf.file,
                kind=mf.kind,
                abs_index=mf.abs_index,
                n_bytes_file=size,
                hash=h,
                n_tensors=mf.n_tensors,
                n_bytes_data=mf.n_bytes_data,
                tensors=mf.tensors,
            )
        )

    return Manifest(
        format=m.format,
        source=m.source,
        files=enriched_files,
        hash_algo="blake2b-128",
        hash_scope=hash_scope,
    )
