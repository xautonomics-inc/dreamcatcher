"""Deterministic stage window planner and feasibility engine.

Computes contiguous stage windows across inference nodes based on
VRAM and disk capacities, performs explicit feasibility arithmetic,
and renders runnable launch flags.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .engine import _normalize_manifest, files_for_window, precheck
from .models import Manifest, NodeCapacity, PartitionPlan, StagePlan

_BYTE_UNITS = {
    "": 1,
    "b": 1,
    "k": 1024,
    "kb": 1000,
    "kib": 1024,
    "m": 1024**2,
    "mb": 1000**2,
    "mib": 1024**2,
    "g": 1024**3,
    "gb": 1000**3,
    "gib": 1024**3,
    "t": 1024**4,
    "tb": 1000**4,
    "tib": 1024**4,
}


def parse_byte_size(value: str | int | float) -> int:
    """Parse a human-readable size string (e.g. '24GB', '16GiB', '500M') into integer bytes."""
    if isinstance(value, (int, float)):
        return int(value)
    s = value.strip()
    if not s:
        raise ValueError("Empty byte size string")
    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z]*)$", s)
    if not m:
        raise ValueError(f"Invalid byte size specification: {value!r}")
    num_str, unit = m.group(1), m.group(2).lower()
    num = float(num_str)
    if unit not in _BYTE_UNITS:
        raise ValueError(f"Unknown size unit: {unit!r} in {value!r}")
    return int(num * _BYTE_UNITS[unit])


def format_byte_size(n_bytes: int, binary: bool = True) -> str:
    """Format an integer byte count into a human-readable string."""
    if n_bytes < 0:
        return f"-{format_byte_size(-n_bytes, binary=binary)}"
    if binary:
        suffixes = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
        base = 1024.0
    else:
        suffixes = ["B", "KB", "MB", "GB", "TB", "PB"]
        base = 1000.0

    val = float(n_bytes)
    for suffix in suffixes:
        if abs(val) < base or suffix == suffixes[-1]:
            if suffix == "B":
                return f"{int(val)} B"
            return f"{val:.2f} {suffix}"
        val /= base
    return f"{val:.2f} {suffixes[-1]}"


def parse_node_spec(spec: str) -> NodeCapacity:
    """Parse node capacity specification string.

    Format: '<name>:<vram>:<disk_free>[:<port>][:<reserve>]'
    Example: 'gpu-a:24GiB:100GiB:8080' or 'gpu-b:16GB:80GB::2GiB' or 'node0:24GiB:100GiB:8080:4GiB'
    """
    parts = spec.strip().split(":")
    if len(parts) not in (3, 4, 5):
        raise ValueError(
            f"Invalid node specification: {spec!r}. "
            "Expected format: <name>:<vram>:<disk_free>[:<port>][:<reserve>] "
            "(e.g. 'node0:24GiB:100GiB:8080' or 'node0:24GiB:100GiB:8080:4GiB')"
        )
    name = parts[0].strip()
    if not name:
        raise ValueError("Node name cannot be empty")
    vram = parse_byte_size(parts[1])
    disk = parse_byte_size(parts[2])
    port = int(parts[3].strip()) if len(parts) >= 4 and parts[3].strip() else None
    reserve = parse_byte_size(parts[4].strip()) if len(parts) == 5 and parts[4].strip() else 0
    return NodeCapacity(
        name=name,
        vram_bytes=vram,
        disk_free_bytes=disk,
        port=port,
        vram_reserve_bytes=reserve,
    )


def partition_layers(
    block_count: int,
    nodes: Sequence[NodeCapacity],
    base_vram: int = 0,
) -> list[tuple[int, int] | None]:
    """Deterministically partition block_count layers across nodes proportional to VRAM capacity.

    Uses the largest-remainder method (Hamilton-Hare) to produce contiguous
    window slices [0, B_0), [B_0, B_1), ..., [B_{n-1}, block_count) with zero gaps.
    Accounts for base weights and per-node VRAM reserves.
    """
    if not nodes:
        raise ValueError("At least one node must be provided for partitioning")
    if block_count <= 0:
        return [None for _ in nodes]
    if len(nodes) == 1:
        return [(0, block_count)]

    # Calculate effective VRAM capacity after subtracting shared base weights and node reserves
    eff_vram = [max(0, n.vram_bytes - base_vram - n.vram_reserve_bytes) for n in nodes]
    total_eff = sum(eff_vram)

    if total_eff == 0:
        # If no node exceeds base_vram + reserve, distribute based on raw available VRAM
        raw_vram = [max(0, n.vram_bytes - n.vram_reserve_bytes) for n in nodes]
        total_raw = sum(raw_vram)
        if total_raw == 0:
            # Fall back to equal partition
            shares = [1.0 / len(nodes) for _ in nodes]
        else:
            shares = [r / total_raw for r in raw_vram]
    else:
        shares = [eff / total_eff for eff in eff_vram]

    targets = [block_count * s for s in shares]
    counts = [int(math.floor(t)) for t in targets]
    remainder = block_count - sum(counts)

    # Sort remaining layer priority by largest fractional share (tie-breaker: node index ascending)
    priority = sorted(
        range(len(nodes)),
        key=lambda idx: (targets[idx] - counts[idx], -idx),
        reverse=True,
    )
    for i in range(remainder):
        counts[priority[i]] += 1

    windows: list[tuple[int, int] | None] = []
    curr = 0
    for c in counts:
        if c > 0:
            windows.append((curr, curr + c))
            curr += c
        else:
            windows.append(None)

    return windows


def plan_stages(
    manifest: dict[str, Any] | Manifest | Path | str,
    nodes: Sequence[NodeCapacity],
    windows: Mapping[str, tuple[int, int] | None] | None = None,
    model_dir: Path | str | None = None,
    alias: str | None = None,
    port_base: int | None = None,
    ctx_size: int | None = None,
    command_binary: str = "llama-stage-runner",
) -> PartitionPlan:
    """Evaluate or generate stage window allocations and compute feasibility arithmetic."""
    if not nodes:
        raise ValueError("At least one node must be provided")

    m = _normalize_manifest(manifest)
    total_blocks = m.source.block_count
    model_name = m.source.model_name

    # Determine default model_dir string for launch flags
    if model_dir is not None:
        model_dir_str = str(model_dir)
    elif isinstance(manifest, (str, Path)):
        p = Path(manifest)
        model_dir_str = str(p.parent if p.is_file() else p)
    else:
        model_dir_str = "<model-dir>"

    # Compute weight per block and shared loader weights
    block_weights: dict[int, int] = {}
    embd_weight = 0
    output_weight = 0
    other_weight = 0

    for mf in m.files:
        if mf.kind in ("layer", "nextn") and mf.abs_index is not None:
            block_weights[mf.abs_index] = mf.n_bytes_data or mf.n_bytes_file
        elif mf.kind == "embd":
            embd_weight = mf.n_bytes_data or mf.n_bytes_file
        elif mf.kind == "output":
            output_weight = mf.n_bytes_data or mf.n_bytes_file
        elif mf.kind == "other":
            other_weight += mf.n_bytes_data or mf.n_bytes_file

    base_vram = embd_weight + output_weight + other_weight

    # Resolve stage windows: either caller supplied or auto-partitioned
    resolved_windows: list[tuple[int, int] | None] = []

    if windows is not None:
        for n in nodes:
            resolved_windows.append(windows.get(n.name))
    else:
        resolved_windows = partition_layers(total_blocks, nodes, base_vram=base_vram)

    # Identify active node indices in sequence
    active_indices = [idx for idx, win in enumerate(resolved_windows) if win is not None]
    n_active = len(active_indices)

    stage_plans: list[StagePlan] = []
    assigned_layer_set: set[int] = set()

    for idx, (node, win) in enumerate(zip(nodes, resolved_windows, strict=True)):
        # Resolve port
        port = node.port
        if port is None and port_base is not None:
            port = port_base + idx

        if win is None:
            stage_plans.append(
                StagePlan(
                    node=node.name,
                    window=None,
                    n_layers=0,
                    files=[],
                    disk_required_bytes=0,
                    disk_free_bytes=node.disk_free_bytes,
                    disk_surplus_bytes=node.disk_free_bytes,
                    disk_deficit_bytes=0,
                    disk_feasible=True,
                    disk_reason="No stage window assigned",
                    vram_required_bytes=0,
                    vram_capacity_bytes=node.vram_bytes,
                    vram_surplus_bytes=node.vram_bytes,
                    vram_deficit_bytes=0,
                    vram_feasible=True,
                    vram_reason="No stage window assigned",
                    vram_reserve_bytes=node.vram_reserve_bytes,
                    vram_basis=(
                        "weights only (layers + parts); KV cache and compute buffers not modelled"
                    ),
                    feasible=True,
                    assembly_flags=[],
                    launch_flags=[],
                    command="",
                )
            )
            continue

        a, b = win
        if a < 0 or b > total_blocks or a >= b:
            raise ValueError(
                f"Invalid window [{a}, {b}) for node {node.name!r}: model has {total_blocks} blocks"
            )

        n_layers = b - a
        assigned_layer_set.update(range(a, b))

        needed_files = sorted(files_for_window(m, a, b))

        # Disk feasibility via precheck()
        pc = precheck(needed_files, m, node.disk_free_bytes)
        disk_required = pc.required_bytes
        disk_free = pc.free_bytes
        disk_surplus = pc.surplus_bytes
        disk_deficit = pc.deficit_bytes
        disk_feasible = pc.feasible
        disk_reason = pc.reason

        # VRAM feasibility (weights + mandatory parts + optional explicit reserve)
        layer_vram = sum(block_weights.get(i, 0) for i in range(a, b))
        reserve_vram = node.vram_reserve_bytes
        vram_required = layer_vram + base_vram + reserve_vram
        vram_capacity = node.vram_bytes
        vram_feasible = vram_capacity >= vram_required
        vram_surplus = max(0, vram_capacity - vram_required)
        vram_deficit = max(0, vram_required - vram_capacity)

        if reserve_vram > 0:
            vram_basis = (
                "weights only (layers + parts) + explicit reserve; "
                "KV cache and compute buffers not dynamically modelled"
            )
            weights_fmt = format_byte_size(layer_vram + base_vram)
            reserve_fmt = format_byte_size(reserve_vram)
            req_fmt = format_byte_size(vram_required)
            cap_fmt = format_byte_size(vram_capacity)
            if vram_feasible:
                surp_fmt = format_byte_size(vram_surplus)
                vram_reason = (
                    f"Feasible (weights+reserve): requires {req_fmt} "
                    f"(weights {weights_fmt} + reserve {reserve_fmt}), "
                    f"capacity {cap_fmt} (surplus {surp_fmt}) "
                    f"[{vram_basis}]"
                )
            else:
                short_fmt = format_byte_size(vram_deficit)
                vram_reason = (
                    f"VRAM deficit (weights+reserve): requires {req_fmt} "
                    f"(weights {weights_fmt} + reserve {reserve_fmt}), "
                    f"capacity {cap_fmt} (shortfall {short_fmt}) "
                    f"[{vram_basis}]"
                )
        else:
            vram_basis = "weights only (layers + parts); KV cache and compute buffers not modelled"
            if vram_feasible:
                vram_reason = (
                    f"Feasible (weights only): requires {format_byte_size(vram_required)}, "
                    f"capacity {format_byte_size(vram_capacity)} "
                    f"(surplus {format_byte_size(vram_surplus)}) "
                    f"[{vram_basis}]"
                )
            else:
                vram_reason = (
                    f"VRAM deficit: requires {format_byte_size(vram_required)}, "
                    f"capacity {format_byte_size(vram_capacity)} "
                    f"(shortfall {format_byte_size(vram_deficit)}) "
                    f"[{vram_basis}]"
                )

        # Model loader assembly flags
        assembly_flags = ["--model-dir", model_dir_str, "--layers", f"{a},{b}"]

        # Active stage position in pipeline
        active_pos = active_indices.index(idx)

        # Launch flags
        if command_binary == "llama-stage-runner":
            if n_active == 1:
                # Single stage: standalone assembly
                flags = list(assembly_flags)
                if port is not None:
                    flags.extend(["--port", str(port)])
                if ctx_size is not None:
                    flags.extend(["--n-ctx", str(ctx_size)])
            else:
                # Multi-stage pipeline wiring: head -> relay -> ... -> tail
                if active_pos == 0:
                    # Head stage connects to next stage
                    next_node = nodes[active_indices[1]]
                    next_port = (
                        next_node.port
                        if next_node.port is not None
                        else (port_base + active_indices[1] if port_base is not None else 8081)
                    )
                    next_host = next_node.host or next_node.name
                    flags = [
                        "--role",
                        "head",
                        "--connect",
                        f"{next_host}:{next_port}",
                    ] + list(assembly_flags)
                    if ctx_size is not None:
                        flags.extend(["--n-ctx", str(ctx_size)])
                elif active_pos < n_active - 1:
                    # Intermediate relay stage: listens and connects downstream
                    my_port = (
                        port
                        if port is not None
                        else (port_base + idx if port_base is not None else 8080 + active_pos)
                    )
                    next_node = nodes[active_indices[active_pos + 1]]
                    next_port = (
                        next_node.port
                        if next_node.port is not None
                        else (
                            port_base + active_indices[active_pos + 1]
                            if port_base is not None
                            else 8080 + active_pos + 1
                        )
                    )
                    next_host = next_node.host or next_node.name
                    flags = [
                        "--role",
                        "relay",
                        "--listen",
                        str(my_port),
                        "--connect",
                        f"{next_host}:{next_port}",
                    ] + list(assembly_flags)
                    if ctx_size is not None:
                        flags.extend(["--n-ctx", str(ctx_size)])
                else:
                    # Tail stage: listens for upstream connection
                    my_port = (
                        port
                        if port is not None
                        else (port_base + idx if port_base is not None else 8080 + active_pos)
                    )
                    flags = [
                        "--role",
                        "tail",
                        "--listen",
                        str(my_port),
                    ] + list(assembly_flags)
                    if ctx_size is not None:
                        flags.extend(["--n-ctx", str(ctx_size)])
        else:
            flags = list(assembly_flags)
            if port is not None:
                flags.extend(["--port", str(port)])
            if alias is not None:
                flags.extend(["--alias", alias])
            if ctx_size is not None:
                flags.extend(["--ctx-size", str(ctx_size)])

        command = f"{command_binary} {' '.join(flags)}"
        stage_feasible = disk_feasible and vram_feasible

        stage_plans.append(
            StagePlan(
                node=node.name,
                window=(a, b),
                n_layers=n_layers,
                files=needed_files,
                disk_required_bytes=disk_required,
                disk_free_bytes=disk_free,
                disk_surplus_bytes=disk_surplus,
                disk_deficit_bytes=disk_deficit,
                disk_feasible=disk_feasible,
                disk_reason=disk_reason,
                vram_required_bytes=vram_required,
                vram_capacity_bytes=vram_capacity,
                vram_surplus_bytes=vram_surplus,
                vram_deficit_bytes=vram_deficit,
                vram_feasible=vram_feasible,
                vram_reason=vram_reason,
                vram_reserve_bytes=reserve_vram,
                vram_basis=vram_basis,
                feasible=stage_feasible,
                assembly_flags=assembly_flags,
                launch_flags=flags,
                command=command,
            )
        )

    all_layers = set(range(total_blocks))
    unassigned = sorted(all_layers - assigned_layer_set)
    all_stages_feasible = all(s.feasible for s in stage_plans)
    coverage_complete = len(unassigned) == 0

    overall_feasible = all_stages_feasible and coverage_complete

    if overall_feasible:
        reason = (
            f"All {total_blocks} layers assigned and feasible across "
            f"{sum(1 for s in stage_plans if s.window is not None)} active nodes"
        )
    elif not coverage_complete:
        reason = f"Incomplete layer coverage: {len(unassigned)} of {total_blocks} layers unassigned"
    else:
        reasons: list[str] = []
        for s in stage_plans:
            if not s.disk_feasible:
                reasons.append(f"{s.node} (disk: {s.disk_reason})")
            if not s.vram_feasible:
                reasons.append(f"{s.node} (vram: {s.vram_reason})")
        reason = f"Infeasible placement on node(s): {', '.join(reasons)}"

    any_reserve = any(n.vram_reserve_bytes > 0 for n in nodes)
    plan_vram_basis = (
        "weights only (layers + parts) + explicit reserve; "
        "KV cache and compute buffers not dynamically modelled"
        if any_reserve
        else "weights only (layers + parts); KV cache and compute buffers not modelled"
    )

    return PartitionPlan(
        feasible=overall_feasible,
        model_name=model_name,
        block_count=total_blocks,
        total_assigned_layers=len(assigned_layer_set),
        unassigned_layers=unassigned,
        stages=stage_plans,
        reason=reason,
        vram_basis=plan_vram_basis,
    )


def render_plan_table(plan: PartitionPlan) -> str:
    """Format partition plan into a structured ASCII table."""
    lines: list[str] = [
        "=== Layer Distribution Stage Plan ===",
        f"Model: {plan.model_name} ({plan.block_count} layers total)",
        f"Status: {'FEASIBLE' if plan.feasible else 'INFEASIBLE'}",
        f"Summary: {plan.reason}",
        f"VRAM Basis: {plan.vram_basis}",
        "",
        f"{'Node':<12} {'Window':<10} {'Layers':<7} {'Disk Req / Free':<22} "
        f"{'VRAM Req / Cap':<22} {'Status':<16} Launch Command",
        "-" * 115,
    ]

    for s in plan.stages:
        win_str = f"[{s.window[0]}, {s.window[1]})" if s.window else "None"
        d_req = format_byte_size(s.disk_required_bytes)
        d_free = format_byte_size(s.disk_free_bytes)
        d_str = f"{d_req} / {d_free}"

        v_req = format_byte_size(s.vram_required_bytes)
        v_cap = format_byte_size(s.vram_capacity_bytes)
        v_str = f"{v_req} / {v_cap}"

        status = "PASS (weights)" if s.feasible else "FAIL"

        lines.append(
            f"{s.node:<12} {win_str:<10} {s.n_layers:<7} {d_str:<22} "
            f"{v_str:<22} {status:<16} {s.command}"
        )

        if not s.feasible:
            if not s.disk_feasible:
                lines.append(f"  └─ Disk Error: {s.disk_reason}")
            if not s.vram_feasible:
                lines.append(f"  └─ VRAM Error: {s.vram_reason}")

    return "\n".join(lines)
