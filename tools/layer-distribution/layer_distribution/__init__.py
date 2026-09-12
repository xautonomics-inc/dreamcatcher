"""GGUF layer-distribution engine.

Pure, deterministic library for layer library distribution over gguf-layer-library/v1.
"""

from typing import TYPE_CHECKING, Any

from .engine import (
    compute_blake2b_128,
    compute_blake2b_128_stream,
    enrich_manifest_hashes,
    files_for_window,
    precheck,
    rebalance,
    verify,
)
from .models import (
    FileStatus,
    FileVerifyResult,
    HashScope,
    Manifest,
    ManifestFile,
    ManifestSource,
    NodeCapacity,
    NodeDelta,
    PartitionPlan,
    PrecheckResult,
    RebalancePlan,
    StagePlan,
    VerifyReport,
)
from .planner import (
    format_byte_size,
    parse_byte_size,
    parse_node_spec,
    partition_layers,
    plan_stages,
    render_plan_table,
)

if TYPE_CHECKING:
    from .slicer import slice_model

__all__ = [
    "FileStatus",
    "FileVerifyResult",
    "HashScope",
    "Manifest",
    "ManifestFile",
    "ManifestSource",
    "NodeCapacity",
    "NodeDelta",
    "PartitionPlan",
    "PrecheckResult",
    "RebalancePlan",
    "StagePlan",
    "VerifyReport",
    "compute_blake2b_128",
    "compute_blake2b_128_stream",
    "enrich_manifest_hashes",
    "files_for_window",
    "format_byte_size",
    "parse_byte_size",
    "parse_node_spec",
    "partition_layers",
    "plan_stages",
    "precheck",
    "rebalance",
    "render_plan_table",
    "slice_model",
    "verify",
]


def __getattr__(name: str) -> Any:
    """Lazy export of slicer components (PEP 562) to avoid importing numpy/gguf unless needed."""
    if name == "slice_model":
        from .slicer import slice_model

        return slice_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Support discovery of lazy attributes."""
    return list(__all__)
