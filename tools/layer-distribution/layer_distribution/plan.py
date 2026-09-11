"""CLI entrypoint for stage window planning and feasibility evaluation.

Usage:
    python3 -m layer_distribution.plan <manifest.json> \
        --node gpu-a:24GiB:100GiB:8080 \
        --node gpu-b:16GiB:80GiB:8081 \
        [--alias gemma4-12b] \
        [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from .models import NodeCapacity
from .planner import (
    parse_byte_size,
    parse_node_spec,
    plan_stages,
    render_plan_table,
)


def parse_window_arg(arg: str) -> tuple[str, tuple[int, int] | None]:
    """Parse a window override argument: 'node:A,B' or 'node:none'."""
    parts = arg.split(":", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid window override: {arg!r}. "
            "Expected format: <node>:<start>,<end> or <node>:none"
        )
    node, win_str = parts[0].strip(), parts[1].strip()
    if win_str.lower() in ("none", "null", "empty"):
        return node, None
    bounds = win_str.split(",")
    if len(bounds) != 2:
        raise ValueError(
            f"Invalid window bounds: {win_str!r} for node {node!r}. Expected '<start>,<end>'"
        )
    return node, (int(bounds[0].strip()), int(bounds[1].strip()))


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python3 -m layer_distribution.plan",
        description="Compute deterministic stage windows and evaluate disk/VRAM feasibility.",
    )
    parser.add_argument(
        "manifest",
        type=str,
        help="Path to manifest.json or layer library directory containing manifest.json.",
    )
    parser.add_argument(
        "--node",
        dest="nodes",
        action="append",
        default=[],
        help=(
            "Node specification: <name>:<vram>:<disk_free>[:<port>][:<reserve>] "
            "(e.g. 'gpu-a:24GiB:100GiB:8080' or 'gpu-b:16GB:80GB::2GiB')"
        ),
    )
    parser.add_argument(
        "--nodes-json",
        type=str,
        default=None,
        help=(
            "Path to JSON file containing list or mapping of node specifications "
            "(use '-' for stdin)."
        ),
    )
    parser.add_argument(
        "--vram-reserve",
        dest="vram_reserves",
        action="append",
        default=[],
        help=(
            "Per-node VRAM reserve override: <node>=<bytes> (e.g. 'gpu-a=4GiB' or 'gpu-b=2000MB')."
        ),
    )
    parser.add_argument(
        "--window",
        dest="windows",
        action="append",
        default=[],
        help="Explicit window override: <node>:<start>,<end> (e.g. 'gpu-a:0,24' or 'gpu-b:none')",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Model directory path to render in launch flags.",
    )
    parser.add_argument(
        "--alias",
        type=str,
        default=None,
        help="Model alias to render in launch flags.",
    )
    parser.add_argument(
        "--port-base",
        type=int,
        default=None,
        help="Base port number for auto-assigning sequential ports (e.g. 8080 -> 8080, 8081).",
    )
    parser.add_argument(
        "--ctx-size",
        type=int,
        default=None,
        help="Context length to render in launch flags (--ctx-size).",
    )
    parser.add_argument(
        "--binary",
        type=str,
        default="llama-stage-runner",
        help="Binary name for rendered launch command (default: 'llama-stage-runner').",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON output instead of human-readable table.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Main CLI execution entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    nodes: list[NodeCapacity] = []

    # Parse --node flags
    for spec in args.nodes:
        try:
            nodes.append(parse_node_spec(spec))
        except ValueError as e:
            sys.stderr.write(f"error: {e}\n")
            return 2

    # Parse --nodes-json if provided
    if args.nodes_json:
        try:
            if args.nodes_json == "-":
                raw_json = sys.stdin.read()
            else:
                raw_json = Path(args.nodes_json).read_text(encoding="utf-8")
            data: Any = json.loads(raw_json)
            if isinstance(data, list):
                for item in data:
                    nodes.append(NodeCapacity.from_dict(item))
            elif isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, dict):
                        item = dict(v)
                        item.setdefault("name", k)
                        nodes.append(NodeCapacity.from_dict(item))
                    else:
                        raise ValueError(f"Invalid node object for key {k!r}")
            else:
                raise ValueError("JSON must contain an array or object of node specs")
        except Exception as e:
            sys.stderr.write(f"error parsing --nodes-json: {e}\n")
            return 2

    if not nodes:
        sys.stderr.write("error: at least one node must be provided via --node or --nodes-json\n")
        return 2

    # Apply --vram-reserve overrides
    for r_spec in args.vram_reserves:
        if "=" not in r_spec:
            sys.stderr.write(
                f"error: invalid --vram-reserve specification: {r_spec!r}."
                " Expected <node>=<bytes>\n"
            )
            return 2
        r_node, r_size = r_spec.split("=", 1)
        r_node = r_node.strip()
        try:
            r_bytes = parse_byte_size(r_size.strip())
        except ValueError as e:
            sys.stderr.write(f"error: invalid reserve size in --vram-reserve {r_spec!r}: {e}\n")
            return 2

        matched = False
        for idx, n in enumerate(nodes):
            if n.name == r_node:
                nodes[idx] = replace(n, vram_reserve_bytes=r_bytes)
                matched = True
        if not matched:
            sys.stderr.write(
                f"error: node {r_node!r} from --vram-reserve not found in declared nodes\n"
            )
            return 2

    # Parse window overrides if provided
    windows: dict[str, tuple[int, int] | None] | None = None
    if args.windows:
        windows = {}
        for w_arg in args.windows:
            try:
                node, win = parse_window_arg(w_arg)
                windows[node] = win
            except ValueError as e:
                sys.stderr.write(f"error: {e}\n")
                return 2

    # Resolve manifest path
    manifest_path = Path(args.manifest)
    if manifest_path.is_dir():
        cand = manifest_path / "manifest.json"
        if cand.is_file():
            manifest_path = cand

    try:
        plan = plan_stages(
            manifest=manifest_path,
            nodes=nodes,
            windows=windows,
            model_dir=args.model_dir,
            alias=args.alias,
            port_base=args.port_base,
            ctx_size=args.ctx_size,
            command_binary=args.binary,
        )
    except Exception as e:
        sys.stderr.write(f"error: {e}\n")
        return 1

    if args.json:
        print(json.dumps(plan.to_dict(), indent=2))
    else:
        print(render_plan_table(plan))

    return 0 if plan.feasible else 1


if __name__ == "__main__":
    sys.exit(main())
