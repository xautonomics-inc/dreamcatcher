"""Data structures and types for the GGUF layer-distribution engine."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class HashScope(str, Enum):
    """Manifest hash scope definitions."""

    WHOLE_FILE = "whole-file"
    TENSOR_AGGREGATE = "tensor-aggregate"


class FileStatus(str, Enum):
    """Per-file verification status outcomes."""

    PASS = "pass"
    FAIL = "fail"
    MISSING = "missing"
    SIZE_MISMATCH = "size-mismatch"
    UNVERIFIED_NO_HASH = "unverified-no-hash"
    UNVERIFIED_SCOPE_MISMATCH = "unverified-scope-mismatch"


@dataclass(frozen=True)
class FileVerifyResult:
    """Individual file verification outcome."""

    file: str
    status: str
    expected_size: int
    actual_size: int | None = None
    expected_hash: str | None = None
    actual_hash: str | None = None
    detail: str = ""
    expected_scope: str | None = None
    actual_scope: str | None = None

    @property
    def passed(self) -> bool:
        """Return True if file has no detectable errors.

        Includes pass, unverified-no-hash, and unverified-scope-mismatch.
        """
        return self.status in (
            FileStatus.PASS.value,
            FileStatus.UNVERIFIED_NO_HASH.value,
            FileStatus.UNVERIFIED_SCOPE_MISMATCH.value,
        )

    @property
    def hash_verified(self) -> bool:
        """Return True if file was cryptographically verified against manifest hash in scope."""
        return self.status == FileStatus.PASS.value

    def to_dict(self) -> dict[str, Any]:
        """Convert result to standard dictionary."""
        return {
            "file": self.file,
            "status": self.status,
            "expected_size": self.expected_size,
            "actual_size": self.actual_size,
            "expected_hash": self.expected_hash,
            "actual_hash": self.actual_hash,
            "expected_scope": self.expected_scope,
            "actual_scope": self.actual_scope,
            "detail": self.detail,
            "passed": self.passed,
            "hash_verified": self.hash_verified,
        }


@dataclass
class VerifyReport:
    """Consolidated verification report across files."""

    passed: bool
    hash_verified: bool
    files: dict[str, FileVerifyResult] = field(default_factory=dict)
    summary: dict[str, int] = field(default_factory=dict)
    hash_scope: str | None = None
    verified_scope: str | None = None

    @property
    def scope_verified(self) -> bool:
        """Return True if hashes were verified matching declared manifest scope."""
        return (
            self.hash_verified
            and self.verified_scope is not None
            and self.hash_scope == self.verified_scope
        )

    def __getitem__(self, filename: str) -> FileVerifyResult:
        """Allow direct key lookup of file results."""
        return self.files[filename]

    def __contains__(self, filename: str) -> bool:
        """Check if filename exists in report."""
        return filename in self.files

    def __iter__(self) -> Any:
        """Iterate over filenames in report."""
        return iter(self.files)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to nested dictionary."""
        return {
            "passed": self.passed,
            "hash_verified": self.hash_verified,
            "scope_verified": self.scope_verified,
            "hash_scope": self.hash_scope,
            "verified_scope": self.verified_scope,
            "summary": dict(self.summary),
            "files": {k: v.to_dict() for k, v in self.files.items()},
        }


@dataclass(frozen=True)
class PrecheckResult:
    """Disk space precheck feasibility outcome with explicit arithmetic."""

    feasible: bool
    required_bytes: int
    free_bytes: int
    deficit_bytes: int
    surplus_bytes: int
    reason: str
    per_file_bytes: dict[str, int] = field(default_factory=dict)

    def __bool__(self) -> bool:
        """Boolean value equals feasibility."""
        return self.feasible

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "feasible": self.feasible,
            "required_bytes": self.required_bytes,
            "free_bytes": self.free_bytes,
            "deficit_bytes": self.deficit_bytes,
            "surplus_bytes": self.surplus_bytes,
            "reason": self.reason,
            "per_file_bytes": dict(self.per_file_bytes),
        }


@dataclass(frozen=True)
class NodeDelta:
    """Delta calculation for a single node across old and new window assignments."""

    node: str
    old_window: tuple[int, int] | None
    new_window: tuple[int, int] | None
    files_to_add: list[str]
    files_removable: list[str]
    bytes_to_add: int
    bytes_removable: int

    def to_dict(self) -> dict[str, Any]:
        """Convert delta to dictionary representation."""
        return {
            "node": self.node,
            "old_window": list(self.old_window) if self.old_window else None,
            "new_window": list(self.new_window) if self.new_window else None,
            "files_to_add": list(self.files_to_add),
            "files_removable": list(self.files_removable),
            "bytes_to_add": self.bytes_to_add,
            "bytes_removable": self.bytes_removable,
        }


@dataclass
class RebalancePlan:
    """Fleet-wide rebalance plan capturing per-node deltas and aggregated totals."""

    nodes: dict[str, NodeDelta] = field(default_factory=dict)
    total_bytes_to_add: int = 0
    total_bytes_removable: int = 0

    def __getitem__(self, node: str) -> NodeDelta:
        """Allow node delta lookup by node identifier."""
        return self.nodes[node]

    def __contains__(self, node: str) -> bool:
        """Check if node identifier exists in rebalance plan."""
        return node in self.nodes

    def __iter__(self) -> Any:
        """Iterate over node identifiers."""
        return iter(self.nodes)

    def to_dict(self) -> dict[str, Any]:
        """Convert rebalance plan to nested dictionary."""
        return {
            "total_bytes_to_add": self.total_bytes_to_add,
            "total_bytes_removable": self.total_bytes_removable,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
        }


@dataclass(frozen=True)
class NodeCapacity:
    """Declared resource capacities for a candidate inference node."""

    name: str
    vram_bytes: int
    disk_free_bytes: int
    port: int | None = None
    host: str | None = None
    vram_reserve_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert node capacity to dictionary representation."""
        return {
            "name": self.name,
            "vram_bytes": self.vram_bytes,
            "disk_free_bytes": self.disk_free_bytes,
            "port": self.port,
            "host": self.host,
            "vram_reserve_bytes": self.vram_reserve_bytes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NodeCapacity:
        """Construct NodeCapacity from dictionary representation."""
        return cls(
            name=str(data["name"]),
            vram_bytes=int(data["vram_bytes"]),
            disk_free_bytes=int(data["disk_free_bytes"]),
            port=int(data["port"]) if data.get("port") is not None else None,
            host=str(data["host"]) if data.get("host") is not None else None,
            vram_reserve_bytes=int(data.get("vram_reserve_bytes", 0) or 0),
        )


@dataclass(frozen=True)
class StagePlan:
    """Planned stage window assignment and feasibility screen for a single node."""

    node: str
    window: tuple[int, int] | None
    n_layers: int
    files: list[str]
    disk_required_bytes: int
    disk_free_bytes: int
    disk_surplus_bytes: int
    disk_deficit_bytes: int
    disk_feasible: bool
    disk_reason: str
    vram_required_bytes: int
    vram_capacity_bytes: int
    vram_surplus_bytes: int
    vram_deficit_bytes: int
    vram_feasible: bool
    vram_reason: str
    vram_reserve_bytes: int = 0
    vram_basis: str = ""
    feasible: bool = False
    assembly_flags: list[str] = field(default_factory=list)
    launch_flags: list[str] = field(default_factory=list)
    command: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert stage plan to dictionary representation."""
        return {
            "node": self.node,
            "window": list(self.window) if self.window else None,
            "n_layers": self.n_layers,
            "files": list(self.files),
            "disk": {
                "required_bytes": self.disk_required_bytes,
                "free_bytes": self.disk_free_bytes,
                "surplus_bytes": self.disk_surplus_bytes,
                "deficit_bytes": self.disk_deficit_bytes,
                "feasible": self.disk_feasible,
                "reason": self.disk_reason,
            },
            "vram": {
                "required_bytes": self.vram_required_bytes,
                "capacity_bytes": self.vram_capacity_bytes,
                "surplus_bytes": self.vram_surplus_bytes,
                "deficit_bytes": self.vram_deficit_bytes,
                "reserve_bytes": self.vram_reserve_bytes,
                "basis": self.vram_basis,
                "feasible": self.vram_feasible,
                "reason": self.vram_reason,
            },
            "feasible": self.feasible,
            "assembly_flags": list(self.assembly_flags),
            "launch_flags": list(self.launch_flags),
            "command": self.command,
        }


@dataclass
class PartitionPlan:
    """Fleet-wide multi-node partition plan with feasibility arithmetic."""

    feasible: bool
    model_name: str
    block_count: int
    total_assigned_layers: int
    unassigned_layers: list[int]
    stages: list[StagePlan]
    reason: str
    vram_basis: str = "weights only (layers + parts); KV cache and compute buffers not modelled"

    def __bool__(self) -> bool:
        """Boolean value equals overall plan feasibility."""
        return self.feasible

    def __getitem__(self, node: str) -> StagePlan:
        """Allow stage plan lookup by node identifier."""
        for s in self.stages:
            if s.node == node:
                return s
        raise KeyError(node)

    def to_dict(self) -> dict[str, Any]:
        """Convert partition plan to dictionary representation."""
        return {
            "feasible": self.feasible,
            "model_name": self.model_name,
            "block_count": self.block_count,
            "total_assigned_layers": self.total_assigned_layers,
            "unassigned_layers": list(self.unassigned_layers),
            "reason": self.reason,
            "vram_basis": self.vram_basis,
            "stages": [s.to_dict() for s in self.stages],
        }


@dataclass(frozen=True)
class ManifestSource:
    """Model source metadata recorded in manifest."""

    model_name: str
    arch: str
    block_count: int
    leading_dense_block_count: int = 0
    nextn_predict_layers: int = 0


@dataclass(frozen=True)
class ManifestFile:
    """Individual file descriptor entry in manifest.json."""

    file: str
    kind: str
    abs_index: int | None
    n_bytes_file: int
    hash: str | None = None
    n_tensors: int = 0
    n_bytes_data: int = 0
    tensors: tuple[dict[str, Any], ...] = ()


@dataclass
class Manifest:
    """Typed representation of a gguf-layer-library/v1 manifest."""

    format: str
    source: ManifestSource
    files: list[ManifestFile]
    hash_algo: str | None = "blake2b-128"
    hash_scope: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Manifest:
        """Parse dictionary structure into typed Manifest."""
        fmt = str(data.get("format", "gguf-layer-library/v1"))
        src_data = data.get("source", {})
        source = ManifestSource(
            model_name=str(src_data.get("model_name", "")),
            arch=str(src_data.get("arch", "")),
            block_count=int(src_data.get("block_count", 0)),
            leading_dense_block_count=int(src_data.get("leading_dense_block_count", 0)),
            nextn_predict_layers=int(src_data.get("nextn_predict_layers", 0)),
        )
        files_data = data.get("files", [])
        files: list[ManifestFile] = []
        for f in files_data:
            tensors_raw = f.get("tensors", [])
            tensors_tuple: tuple[dict[str, Any], ...] = (
                tuple(dict(t) for t in tensors_raw if isinstance(t, Mapping))
                if isinstance(tensors_raw, list)
                else ()
            )
            files.append(
                ManifestFile(
                    file=str(f["file"]),
                    kind=str(f.get("kind", "layer")),
                    abs_index=int(f["abs_index"]) if f.get("abs_index") is not None else None,
                    n_bytes_file=int(f.get("n_bytes_file", 0)),
                    hash=str(f["hash"]) if f.get("hash") else None,
                    n_tensors=int(f.get("n_tensors", 0)),
                    n_bytes_data=int(f.get("n_bytes_data", 0)),
                    tensors=tensors_tuple,
                )
            )

        hash_algo = data.get("hash_algo", "blake2b-128")

        # Hash scope resolution according to SPEC-020 / meta#62 / meta#65:
        # 1. If "hash_scope" is explicitly declared, use it.
        # 2. If absent:
        #    - Inspect each hashed file vs its tensors[].hash:
        #      * If every file's hash reproduces the blake2b-128 aggregate
        #        of its tensors[].hash -> "tensor-aggregate".
        #      * If file-level hashes are present but per-tensor hashes are absent -> "whole-file".
        #      * If inconsistent or mixed scopes detected -> raise ValueError.
        #    - If no files have hashes: hash_scope = None.
        if "hash_scope" in data:
            raw_scope = data["hash_scope"]
            hash_scope = str(raw_scope) if raw_scope is not None else None
        else:
            file_scopes: set[str] = set()
            for f in files_data:
                file_hash = f.get("hash")
                if not file_hash:
                    continue
                tensors = f.get("tensors")
                if tensors and all(isinstance(t, Mapping) and t.get("hash") for t in tensors):
                    agg = hashlib.blake2b(digest_size=16)
                    for t in tensors:
                        agg.update(bytes.fromhex(str(t["hash"])))
                    if agg.hexdigest() == str(file_hash):
                        file_scopes.add(HashScope.TENSOR_AGGREGATE.value)
                    else:
                        file_scopes.add("invalid-aggregate")
                elif not tensors or all(
                    isinstance(t, Mapping) and not t.get("hash") for t in tensors
                ):
                    file_scopes.add(HashScope.WHOLE_FILE.value)
                else:
                    file_scopes.add("mixed-tensors")

            if not file_scopes:
                hash_scope = None
            elif file_scopes == {HashScope.TENSOR_AGGREGATE.value}:
                hash_scope = HashScope.TENSOR_AGGREGATE.value
            elif file_scopes == {HashScope.WHOLE_FILE.value}:
                hash_scope = HashScope.WHOLE_FILE.value
            else:
                scopes = sorted(file_scopes)
                raise ValueError(
                    f"Inconsistent or mixed hash scopes detected across manifest files: {scopes}"
                )

        return cls(
            format=fmt,
            source=source,
            files=files,
            hash_algo=hash_algo,
            hash_scope=hash_scope,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest back to dictionary structure."""
        return {
            "format": self.format,
            "source": {
                "model_name": self.source.model_name,
                "arch": self.source.arch,
                "block_count": self.source.block_count,
                "leading_dense_block_count": self.source.leading_dense_block_count,
                "nextn_predict_layers": self.source.nextn_predict_layers,
            },
            "files": [
                {
                    "file": f.file,
                    "kind": f.kind,
                    "abs_index": f.abs_index,
                    "n_bytes_file": f.n_bytes_file,
                    "hash": f.hash,
                    "n_tensors": f.n_tensors,
                    "n_bytes_data": f.n_bytes_data,
                    **({"tensors": [dict(t) for t in f.tensors]} if f.tensors else {}),
                }
                for f in self.files
            ],
            "hash_algo": self.hash_algo,
            "hash_scope": self.hash_scope,
        }
