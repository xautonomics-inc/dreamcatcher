"""Tests for layer_distribution stage planner and plan CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from layer_distribution import (
    NodeCapacity,
    format_byte_size,
    parse_byte_size,
    parse_node_spec,
    partition_layers,
    plan_stages,
)
from layer_distribution.plan import main as plan_cli_main


def create_sample_manifest(n_blocks: int = 40, n_nextn: int = 4) -> dict[str, Any]:
    """Build a deterministic sample manifest for planning tests."""
    block_size = 500_000_000  # 500 MB per transformer block
    nextn_size = 300_000_000  # 300 MB per MTP block
    embd_size = 400_000_000  # 400 MB embd
    output_size = 450_000_000  # 450 MB output

    files: list[dict[str, Any]] = []

    for i in range(n_blocks):
        files.append(
            {
                "file": f"blk-{i:05d}.gguf",
                "kind": "layer",
                "abs_index": i,
                "n_bytes_file": block_size,
                "n_bytes_data": block_size - 1024,
                "hash": "0" * 32,
            }
        )

    for i in range(n_blocks, n_blocks + n_nextn):
        files.append(
            {
                "file": f"parts-nextn-{i:05d}.gguf",
                "kind": "nextn",
                "abs_index": i,
                "n_bytes_file": nextn_size,
                "n_bytes_data": nextn_size - 1024,
                "hash": "1" * 32,
            }
        )

    files.append(
        {
            "file": "parts-embd.gguf",
            "kind": "embd",
            "abs_index": None,
            "n_bytes_file": embd_size,
            "n_bytes_data": embd_size - 1024,
            "hash": "2" * 32,
        }
    )

    files.append(
        {
            "file": "parts-output.gguf",
            "kind": "output",
            "abs_index": None,
            "n_bytes_file": output_size,
            "n_bytes_data": output_size - 1024,
            "hash": "3" * 32,
        }
    )

    total_blocks = n_blocks + n_nextn
    return {
        "format": "gguf-layer-library/v1",
        "created": "2026-09-10T00:00:00Z",
        "tool": "slice_gguf_layers.py",
        "hash_algo": "blake2b-128",
        "hash_scope": "whole-file",
        "source": {
            "glob": "models/model.gguf",
            "shards": [{"name": "model.gguf", "n_bytes": 20_000_000_000}],
            "model_name": "test-planner-model",
            "arch": "qwen2",
            "block_count": total_blocks,
            "leading_dense_block_count": 0,
            "nextn_predict_layers": n_nextn,
        },
        "files": files,
    }


def test_parse_byte_size() -> None:
    assert parse_byte_size(1024) == 1024
    assert parse_byte_size("1024") == 1024
    assert parse_byte_size("1024 B") == 1024
    assert parse_byte_size("16 KiB") == 16 * 1024
    assert parse_byte_size("16 KB") == 16 * 1000
    assert parse_byte_size("500 MiB") == 500 * 1024**2
    assert parse_byte_size("500 MB") == 500 * 1000**2
    assert parse_byte_size("24 GiB") == 24 * 1024**3
    assert parse_byte_size("24 GB") == 24 * 1000**3
    assert parse_byte_size("24G") == 24 * 1024**3
    assert parse_byte_size("1.5 TiB") == int(1.5 * 1024**4)

    with pytest.raises(ValueError, match="Empty byte size"):
        parse_byte_size("")
    with pytest.raises(ValueError, match="Invalid byte size"):
        parse_byte_size("invalid")
    with pytest.raises(ValueError, match="Unknown size unit"):
        parse_byte_size("100 foo")


def test_format_byte_size() -> None:
    assert format_byte_size(0) == "0 B"
    assert format_byte_size(512) == "512 B"
    assert format_byte_size(1024) == "1.00 KiB"
    assert format_byte_size(1024**2 * 15) == "15.00 MiB"
    assert format_byte_size(1024**3 * 24) == "24.00 GiB"
    assert format_byte_size(-1024) == "-1.00 KiB"
    assert format_byte_size(1000, binary=False) == "1.00 KB"


def test_parse_node_spec() -> None:
    n1 = parse_node_spec("gpu-a:24GiB:100GiB:8080")
    assert n1.name == "gpu-a"
    assert n1.vram_bytes == 24 * 1024**3
    assert n1.disk_free_bytes == 100 * 1024**3
    assert n1.port == 8080
    assert n1.vram_reserve_bytes == 0

    n2 = parse_node_spec("gpu-b:16GB:80GB")
    assert n2.name == "gpu-b"
    assert n2.vram_bytes == 16 * 1000**3
    assert n2.disk_free_bytes == 80 * 1000**3
    assert n2.port is None
    assert n2.vram_reserve_bytes == 0

    n3 = parse_node_spec("node0:24GiB:100GiB:8080:4GiB")
    assert n3.name == "node0"
    assert n3.port == 8080
    assert n3.vram_reserve_bytes == 4 * 1024**3

    n4 = parse_node_spec("node1:32GiB:200GiB::8GiB")
    assert n4.name == "node1"
    assert n4.port is None
    assert n4.vram_reserve_bytes == 8 * 1024**3

    with pytest.raises(ValueError, match="Expected format"):
        parse_node_spec("too:few")
    with pytest.raises(ValueError, match="Node name cannot be empty"):
        parse_node_spec(":16G:50G")


def test_partition_layers_deterministic() -> None:
    nodes = [
        NodeCapacity(name="nodeA", vram_bytes=24 * 1024**3, disk_free_bytes=100 * 1024**3),
        NodeCapacity(name="nodeB", vram_bytes=16 * 1024**3, disk_free_bytes=100 * 1024**3),
    ]

    # Run partition multiple times to assert strict determinism
    for _ in range(5):
        w = partition_layers(40, nodes)
        assert w == [(0, 24), (24, 40)]

    # Single node gets all layers
    w_single = partition_layers(40, [nodes[0]])
    assert w_single == [(0, 40)]

    # 3 equal nodes
    equal_nodes = [
        NodeCapacity(name=f"n{i}", vram_bytes=16 * 1024**3, disk_free_bytes=50 * 1024**3)
        for i in range(3)
    ]
    w_equal = partition_layers(48, equal_nodes)
    assert w_equal == [(0, 16), (16, 32), (32, 48)]

    # Node with reserve reduces its layer share
    reserved_nodes = [
        NodeCapacity(
            name="n0",
            vram_bytes=24 * 1024**3,
            disk_free_bytes=100 * 1024**3,
            vram_reserve_bytes=8 * 1024**3,
        ),  # 16 GB effective
        NodeCapacity(
            name="n1",
            vram_bytes=16 * 1024**3,
            disk_free_bytes=100 * 1024**3,
            vram_reserve_bytes=0,
        ),  # 16 GB effective
    ]
    w_res = partition_layers(40, reserved_nodes)
    assert w_res == [(0, 20), (20, 40)]

    # Node with 0 effective VRAM receives None
    skewed = [
        NodeCapacity(name="big", vram_bytes=32 * 1024**3, disk_free_bytes=100 * 1024**3),
        NodeCapacity(name="tiny", vram_bytes=1 * 1024**3, disk_free_bytes=100 * 1024**3),
    ]
    w_skewed = partition_layers(30, skewed, base_vram=2 * 1024**3)
    assert w_skewed == [(0, 30), None]


def test_plan_stages_feasible(tmp_path: Path) -> None:
    manifest_data = create_sample_manifest(n_blocks=40, n_nextn=4)
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    nodes = [
        NodeCapacity(
            name="gpu-a",
            vram_bytes=24 * 1024**3,
            disk_free_bytes=100 * 1024**3,
            port=8080,
        ),
        NodeCapacity(
            name="gpu-b",
            vram_bytes=16 * 1024**3,
            disk_free_bytes=80 * 1024**3,
            port=8081,
        ),
    ]

    plan = plan_stages(
        manifest=manifest_file,
        nodes=nodes,
        alias="test-model",
    )

    assert plan.feasible is True
    assert plan.block_count == 44
    assert plan.total_assigned_layers == 44
    assert len(plan.unassigned_layers) == 0
    assert len(plan.stages) == 2
    assert "weights only" in plan.vram_basis

    s0 = plan["gpu-a"]
    assert s0.feasible is True
    assert s0.disk_feasible is True
    assert s0.vram_feasible is True
    assert s0.window is not None
    assert s0.window[0] == 0
    assert s0.assembly_flags == ["--model-dir", str(tmp_path), "--layers", f"0,{s0.window[1]}"]
    assert "--role" in s0.launch_flags
    assert "head" in s0.launch_flags
    assert "--connect" in s0.launch_flags
    assert "gpu-b:8081" in s0.launch_flags
    assert s0.command.startswith("llama-stage-runner --role head --connect gpu-b:8081")

    s1 = plan["gpu-b"]
    assert s1.feasible is True
    assert s1.window is not None
    assert s1.window[1] == 44
    assert s0.window[1] == s1.window[0]
    assert s1.assembly_flags == ["--model-dir", str(tmp_path), "--layers", f"{s1.window[0]},44"]
    assert "--role" in s1.launch_flags
    assert "tail" in s1.launch_flags
    assert "--listen" in s1.launch_flags
    assert "8081" in s1.launch_flags
    assert s1.command.startswith("llama-stage-runner --role tail --listen 8081")


def test_plan_stages_three_stages_pipeline_wiring(tmp_path: Path) -> None:
    manifest_data = create_sample_manifest(n_blocks=30, n_nextn=0)
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    nodes = [
        NodeCapacity(
            name="node0", vram_bytes=24 * 1024**3, disk_free_bytes=100 * 1024**3, port=8080
        ),
        NodeCapacity(
            name="node1", vram_bytes=24 * 1024**3, disk_free_bytes=100 * 1024**3, port=8081
        ),
        NodeCapacity(
            name="node2", vram_bytes=24 * 1024**3, disk_free_bytes=100 * 1024**3, port=8082
        ),
    ]

    plan = plan_stages(manifest=manifest_file, nodes=nodes)
    assert plan.feasible is True
    assert len(plan.stages) == 3

    # Stage 0: head connects to node1:8081
    s0 = plan["node0"]
    assert s0.launch_flags[:4] == ["--role", "head", "--connect", "node1:8081"]
    assert s0.assembly_flags == ["--model-dir", str(tmp_path), "--layers", "0,10"]

    # Stage 1: relay listens on 8081, connects to node2:8082
    s1 = plan["node1"]
    assert s1.launch_flags[:6] == ["--role", "relay", "--listen", "8081", "--connect", "node2:8082"]
    assert s1.assembly_flags == ["--model-dir", str(tmp_path), "--layers", "10,20"]

    # Stage 2: tail listens on 8082
    s2 = plan["node2"]
    assert s2.launch_flags[:4] == ["--role", "tail", "--listen", "8082"]
    assert s2.assembly_flags == ["--model-dir", str(tmp_path), "--layers", "20,30"]


def test_plan_stages_custom_binary(tmp_path: Path) -> None:
    manifest_data = create_sample_manifest(n_blocks=10, n_nextn=0)
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    nodes = [
        NodeCapacity(name="s1", vram_bytes=20 * 1024**3, disk_free_bytes=50 * 1024**3, port=8080),
    ]

    plan = plan_stages(
        manifest=manifest_file,
        nodes=nodes,
        command_binary="llama-server",
        alias="test-alias",
        ctx_size=8192,
    )
    s = plan["s1"]
    assert s.command.startswith("llama-server")
    assert "--alias" in s.launch_flags
    assert "--ctx-size" in s.launch_flags
    assert "--port" in s.launch_flags


def test_plan_stages_vram_reserve(tmp_path: Path) -> None:
    manifest_data = create_sample_manifest(n_blocks=10, n_nextn=0)
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    # Node has 10 GB VRAM. Model weights need ~5.85 GB.
    # Without reserve: feasible. With 6 GB reserve: required ~11.85 GB -> deficit.
    node_tight = NodeCapacity(
        name="tight",
        vram_bytes=10 * 1024**3,
        disk_free_bytes=50 * 1024**3,
        vram_reserve_bytes=6 * 1024**3,
    )
    plan = plan_stages(manifest=manifest_file, nodes=[node_tight])
    assert plan.feasible is False
    s = plan["tight"]
    assert s.vram_feasible is False
    assert s.vram_deficit_bytes > 0
    assert "reserve" in s.vram_basis
    assert "reserve" in s.vram_reason
    assert "KV cache and compute buffers not dynamically modelled" in s.vram_basis


def test_plan_stages_disk_deficit(tmp_path: Path) -> None:
    manifest_data = create_sample_manifest(n_blocks=10, n_nextn=0)
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    nodes = [
        # Only 500 MB disk free; model stage requires > 1 GB
        NodeCapacity(name="cramped", vram_bytes=24 * 1024**3, disk_free_bytes=500_000_000),
    ]

    plan = plan_stages(manifest=manifest_file, nodes=nodes)
    assert plan.feasible is False
    assert plan["cramped"].disk_feasible is False
    assert plan["cramped"].disk_deficit_bytes > 0
    assert "Infeasible placement" in plan.reason


def test_plan_stages_vram_deficit(tmp_path: Path) -> None:
    manifest_data = create_sample_manifest(n_blocks=10, n_nextn=0)
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    nodes = [
        # Only 100 MB VRAM; base alone is ~850 MB
        NodeCapacity(name="low_vram", vram_bytes=100_000_000, disk_free_bytes=100 * 1024**3),
    ]

    plan = plan_stages(manifest=manifest_file, nodes=nodes)
    assert plan.feasible is False
    assert plan["low_vram"].vram_feasible is False
    assert plan["low_vram"].vram_deficit_bytes > 0


def test_plan_stages_custom_windows(tmp_path: Path) -> None:
    manifest_data = create_sample_manifest(n_blocks=20, n_nextn=0)
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    nodes = [
        NodeCapacity(name="n1", vram_bytes=20 * 1024**3, disk_free_bytes=50 * 1024**3),
        NodeCapacity(name="n2", vram_bytes=20 * 1024**3, disk_free_bytes=50 * 1024**3),
    ]

    # Explicit full coverage
    plan = plan_stages(
        manifest=manifest_file,
        nodes=nodes,
        windows={"n1": (0, 10), "n2": (10, 20)},
    )
    assert plan.feasible is True
    assert plan["n1"].window == (0, 10)
    assert plan["n2"].window == (10, 20)

    # Incomplete coverage leaves unassigned layers
    plan_gap = plan_stages(
        manifest=manifest_file,
        nodes=nodes,
        windows={"n1": (0, 5), "n2": (15, 20)},
    )
    assert plan_gap.feasible is False
    assert len(plan_gap.unassigned_layers) == 10
    assert "Incomplete layer coverage" in plan_gap.reason


def test_plan_cli_table_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_data = create_sample_manifest(n_blocks=10, n_nextn=0)
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    rc = plan_cli_main(
        [
            str(manifest_file),
            "--node",
            "gpu-a:24GiB:100GiB:8080",
            "--node",
            "gpu-b:16GiB:80GiB:8081",
            "--alias",
            "gemma4",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "=== Layer Distribution Stage Plan ===" in captured.out
    assert "FEASIBLE" in captured.out
    assert "PASS (weights)" in captured.out
    assert "VRAM Basis:" in captured.out
    assert "gpu-a" in captured.out
    assert "gpu-b" in captured.out
    assert "--layers" in captured.out
    assert "--role head" in captured.out
    assert "--role tail" in captured.out


def test_plan_cli_json_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_data = create_sample_manifest(n_blocks=10, n_nextn=0)
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    rc = plan_cli_main(
        [
            str(manifest_file),
            "--node",
            "gpu-a:24GiB:100GiB:8080:2GiB",
            "--json",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["feasible"] is True
    assert "vram_basis" in data
    assert len(data["stages"]) == 1
    s0 = data["stages"][0]
    assert s0["node"] == "gpu-a"
    assert s0["disk"]["feasible"] is True
    assert s0["vram"]["feasible"] is True
    assert s0["vram"]["reserve_bytes"] == 2 * 1024**3
    assert "basis" in s0["vram"]
    assert "assembly_flags" in s0


def test_plan_cli_vram_reserve_override(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_data = create_sample_manifest(n_blocks=10, n_nextn=0)
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    rc = plan_cli_main(
        [
            str(manifest_file),
            "--node",
            "gpu-a:24GiB:100GiB:8080",
            "--vram-reserve",
            "gpu-a=4GiB",
            "--json",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["stages"][0]["vram"]["reserve_bytes"] == 4 * 1024**3


def test_plan_cli_nodes_json(tmp_path: Path) -> None:
    manifest_data = create_sample_manifest(n_blocks=10, n_nextn=0)
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    nodes_file = tmp_path / "nodes.json"
    nodes_file.write_text(
        json.dumps(
            [
                {
                    "name": "n1",
                    "vram_bytes": 20 * 1024**3,
                    "disk_free_bytes": 50 * 1024**3,
                    "port": 8080,
                },
                {
                    "name": "n2",
                    "vram_bytes": 20 * 1024**3,
                    "disk_free_bytes": 50 * 1024**3,
                    "port": 8081,
                },
            ]
        ),
        encoding="utf-8",
    )

    rc = plan_cli_main(
        [
            str(manifest_file),
            "--nodes-json",
            str(nodes_file),
        ]
    )
    assert rc == 0


def test_plan_cli_infeasible_exit_code(tmp_path: Path) -> None:
    manifest_data = create_sample_manifest(n_blocks=10, n_nextn=0)
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    rc = plan_cli_main(
        [
            str(manifest_file),
            "--node",
            "starved:100MB:100MB",
        ]
    )
    assert rc == 1
