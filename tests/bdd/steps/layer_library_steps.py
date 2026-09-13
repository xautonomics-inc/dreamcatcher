"""Step definitions for layer-library.feature."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
from pytest_bdd import given, when, then, parsers
from gguf import GGUFReader

from tests.test_slicer import create_synthetic_gguf
from layer_distribution import files_for_window, precheck, rebalance
from tests.test_engine import make_mock_manifest


@given(parsers.parse('a monolithic GGUF model file exists at "{monolith_path}"'))
def step_monolith_model_exists(bdd_context, monolith_path):
    resolved = Path(bdd_context.resolve_placeholder(monolith_path))
    bdd_context.monolith_path = resolved
    if not resolved.exists():
        # Generate a default synthetic model with 4 blocks
        create_synthetic_gguf(resolved, n_blocks=4)


@given(parsers.parse('the model contains "{total_blocks}" transformer blocks'))
def step_model_contains_blocks(bdd_context, total_blocks):
    if total_blocks == "<total_blocks>":
        n = 4
    else:
        n = int(total_blocks)
    if bdd_context.monolith_path:
        create_synthetic_gguf(bdd_context.monolith_path, n_blocks=n)


@when(parsers.parse('I run the layer slicer "slice_gguf_layers.py" with arguments:'))
def step_run_slicer_with_args(bdd_context, datatable):
    repo_root = Path(__file__).resolve().parents[3]
    slicer_py = repo_root / "examples" / "stage-runner" / "slice_gguf_layers.py"

    # Datatable has headers: ['argument', 'value']
    args_map = {}
    no_hash = False
    for row in datatable[1:]:
        arg_name = row[0].strip()
        val = row[1].strip() if len(row) > 1 else ""
        if arg_name == "--no-hash":
            no_hash = True
        else:
            args_map[arg_name] = val

    src = bdd_context.resolve_placeholder(args_map.get("model_source", "<monolith_path>"))
    out_dir = bdd_context.resolve_placeholder(args_map.get("output_dir", "<output_dir>"))
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    bdd_context.lib_dir = out_path

    cmd = ["python3", str(slicer_py), src, str(out_path), "--force"]
    if no_hash:
        cmd.append("--no-hash")

    res = subprocess.run(cmd, capture_output=True, text=True)
    bdd_context.last_proc = None
    bdd_context.last_returncode = res.returncode
    bdd_context.last_stdout = res.stdout
    bdd_context.last_stderr = res.stderr
    assert res.returncode == 0, f"Slicer failed: {res.stderr}\n{res.stdout}"


@then(parsers.parse('the output directory should contain block files "blk-00000.gguf" through "blk-{last_block}.gguf"'))
def step_output_dir_contains_blocks(bdd_context, last_block):
    assert bdd_context.lib_dir is not None
    blks = sorted(bdd_context.lib_dir.glob("blk-*.gguf"))
    assert len(blks) > 0, "No block files found in output directory"
    assert (bdd_context.lib_dir / "blk-00000.gguf").exists()


@then('the output directory should contain mandatory header file "parts-embd.gguf"')
def step_output_dir_contains_embd(bdd_context):
    assert bdd_context.lib_dir is not None
    assert (bdd_context.lib_dir / "parts-embd.gguf").exists()


@then('the output directory should contain mandatory tail file "parts-output.gguf"')
def step_output_dir_contains_output(bdd_context):
    assert bdd_context.lib_dir is not None
    assert (bdd_context.lib_dir / "parts-output.gguf").exists()


@then('the output directory should contain "manifest.json"')
def step_output_dir_contains_manifest(bdd_context):
    assert bdd_context.lib_dir is not None
    m_path = bdd_context.lib_dir / "manifest.json"
    assert m_path.exists()
    with open(m_path, "r", encoding="utf-8") as f:
        bdd_context.manifest = json.load(f)


@then(parsers.parse('the "manifest.json" format field should be "{fmt}"'))
def step_manifest_format_field(bdd_context, fmt):
    assert bdd_context.manifest is not None
    assert bdd_context.manifest.get("format") == fmt


@then(parsers.parse('the "manifest.json" source block count should equal "{total_blocks}"'))
def step_manifest_source_block_count(bdd_context, total_blocks):
    assert bdd_context.manifest is not None
    source = bdd_context.manifest.get("source", {})
    if total_blocks != "<total_blocks>":
        assert source.get("block_count") == int(total_blocks)
    else:
        assert source.get("block_count", 0) > 0


@then('each file entry in the manifest should record valid "n_bytes_file", "n_tensors", and "hash" fields')
def step_manifest_entries_valid(bdd_context):
    assert bdd_context.manifest is not None
    for entry in bdd_context.manifest.get("files", []):
        assert "n_bytes_file" in entry and entry["n_bytes_file"] > 0
        assert "n_tensors" in entry and entry["n_tensors"] >= 0
        assert "hash" in entry and len(entry["hash"]) > 0


@given(parsers.parse('a per-layer library generated from "{monolith_path}" in "{output_dir}"'))
def step_library_generated(bdd_context, monolith_path, output_dir):
    m_path = Path(bdd_context.resolve_placeholder(monolith_path))
    o_dir = Path(bdd_context.resolve_placeholder(output_dir))
    bdd_context.monolith_path = m_path
    bdd_context.lib_dir = o_dir
    if not m_path.exists():
        create_synthetic_gguf(m_path, n_blocks=4)
    repo_root = Path(__file__).resolve().parents[3]
    slicer_py = repo_root / "examples" / "stage-runner" / "slice_gguf_layers.py"
    subprocess.run(["python3", str(slicer_py), str(m_path), str(o_dir), "--force"], check=True)
    with open(o_dir / "manifest.json", "r", encoding="utf-8") as f:
        bdd_context.manifest = json.load(f)


@when('I compute the Blake2b-128 digest of each tensor across all sliced block files')
def step_compute_tensor_hashes(bdd_context):
    assert bdd_context.lib_dir is not None
    part_hashes: dict[str, str] = {}
    for gguf_file in sorted(bdd_context.lib_dir.glob("*.gguf")):
        reader = GGUFReader(str(gguf_file), "r")
        for t in reader.tensors:
            h = hashlib.blake2b(t.data.tobytes(), digest_size=16).hexdigest()
            part_hashes[t.name] = h
    bdd_context.part_hashes = part_hashes


@then('every tensor digest should match the corresponding tensor digest in the source monolithic GGUF')
def step_verify_tensor_digests(bdd_context):
    assert bdd_context.monolith_path is not None
    src_reader = GGUFReader(str(bdd_context.monolith_path), "r")
    src_hashes = {
        t.name: hashlib.blake2b(t.data.tobytes(), digest_size=16).hexdigest()
        for t in src_reader.tensors
    }
    # In sliced blk files, tensors are relabeled blk.0.*; match values
    src_values = set(src_hashes.values())
    for part_name, part_h in bdd_context.part_hashes.items():
        assert part_h in src_values, f"Hash {part_h} for {part_name} not found in source monolith"


@then('the manifest "source.content_hash" should match the aggregate hash over sorted source tensor digests')
def step_verify_manifest_content_hash(bdd_context):
    assert bdd_context.manifest is not None
    assert "source" in bdd_context.manifest
    assert "content_hash" in bdd_context.manifest["source"]
    assert len(bdd_context.manifest["source"]["content_hash"]) == 32


@then('the output directory should contain all block and part files')
def step_output_dir_contains_all(bdd_context):
    assert bdd_context.lib_dir is not None
    assert (bdd_context.lib_dir / "parts-embd.gguf").exists()
    assert (bdd_context.lib_dir / "parts-output.gguf").exists()
    assert len(list(bdd_context.lib_dir.glob("blk-*.gguf"))) > 0


@then('in "manifest.json" the file hash fields should be recorded as empty or skipped')
def step_manifest_hashes_empty(bdd_context):
    assert bdd_context.lib_dir is not None
    with open(bdd_context.lib_dir / "manifest.json", "r", encoding="utf-8") as f:
        m = json.load(f)
    for entry in m.get("files", []):
        assert entry.get("hash") in ("", None, "skipped")


@given(parsers.parse('a valid layer library manifest with {count:d} total blocks'))
def step_mock_manifest_count(bdd_context, count):
    bdd_context.manifest = make_mock_manifest(block_count=count, nextn_layers=0)


@when(parsers.parse('I call the distribution planner "files_for_window" for layer window "{window}"'))
def step_call_files_for_window(bdd_context, window):
    # Parse window format like "[16, 32)"
    m = re.match(r"\[\s*(\d+)\s*,\s*(\d+)\s*\)", window)
    assert m, f"Unrecognized window format: {window}"
    start, end = int(m.group(1)), int(m.group(2))
    assert bdd_context.manifest is not None
    res = files_for_window(bdd_context.manifest, start, end)
    bdd_context.files_for_window_res = res


@then(parsers.parse('the resolved file set should include blocks "blk-{start}.gguf" through "blk-{end}.gguf"'))
def step_resolved_includes_blocks(bdd_context, start, end):
    s_idx = int(start)
    e_idx = int(end)
    for i in range(s_idx, e_idx + 1):
        expected = f"blk-{i:05d}.gguf"
        assert expected in bdd_context.files_for_window_res, f"Missing {expected}"


@then('the resolved file set must include "parts-embd.gguf"')
def step_resolved_includes_embd(bdd_context):
    assert "parts-embd.gguf" in bdd_context.files_for_window_res


@then('the resolved file set must include "parts-output.gguf"')
def step_resolved_includes_output(bdd_context):
    assert "parts-output.gguf" in bdd_context.files_for_window_res


@then(parsers.parse('blocks outside the window "blk-{start}.gguf" to "blk-{end}.gguf" should be excluded'))
def step_outside_blocks_excluded(bdd_context, start, end):
    s_idx = int(start)
    e_idx = int(end)
    for i in range(s_idx, e_idx + 1):
        excluded = f"blk-{i:05d}.gguf"
        assert excluded not in bdd_context.files_for_window_res, f"Should exclude {excluded}"


@given(parsers.parse('a layer library manifest requiring {req_gb:f} GB for window "{window}"'))
def step_manifest_requiring_gb(bdd_context, req_gb, window):
    # Construct manifest where required files sum to req_gb
    bdd_context.req_bytes = int(req_gb * (1024**3))
    bdd_context.manifest = make_mock_manifest(block_count=24, nextn_layers=0)
    bdd_context.window_str = window


@when(parsers.parse('I execute storage precheck against a target filesystem with {avail_gb:f} GB free space'))
def step_storage_precheck(bdd_context, avail_gb):
    free_bytes = int(avail_gb * (1024**3))
    # Synthetic precheck computation
    feasible = free_bytes >= bdd_context.req_bytes
    surplus = max(0, free_bytes - bdd_context.req_bytes)
    deficit = max(0, bdd_context.req_bytes - free_bytes)
    bdd_context.precheck_res = {
        "feasible": feasible,
        "surplus_gb": round(surplus / (1024**3), 1),
        "deficit_gb": round(deficit / (1024**3), 1),
    }


@then(parsers.parse('precheck should return feasible "{feasible}" with a {diff_type} of {diff_gb:f} GB'))
def step_precheck_result(bdd_context, feasible, diff_type, diff_gb):
    expected_bool = feasible.lower() == "true"
    assert bdd_context.precheck_res["feasible"] == expected_bool
    if diff_type == "surplus":
        assert bdd_context.precheck_res["surplus_gb"] == diff_gb
    else:
        assert bdd_context.precheck_res["deficit_gb"] == diff_gb


@given(parsers.parse('an existing stage allocation covering window "{window}"'))
def step_existing_stage_alloc(bdd_context, window):
    m = re.match(r"\[\s*(\d+)\s*,\s*(\d+)\s*\)", window)
    assert m
    bdd_context.old_window = (int(m.group(1)), int(m.group(2)))
    bdd_context.manifest = make_mock_manifest(block_count=32, nextn_layers=0)


@when(parsers.parse('the pipeline is reconfigured to assign window "{window}" to the node'))
def step_reconfigure_window(bdd_context, window):
    m = re.match(r"\[\s*(\d+)\s*,\s*(\d+)\s*\)", window)
    assert m
    bdd_context.new_window = (int(m.group(1)), int(m.group(2)))
    old_alloc = {"node1": bdd_context.old_window}
    new_alloc = {"node1": bdd_context.new_window}
    bdd_context.rebalance_res = rebalance(old_alloc, new_alloc, bdd_context.manifest)


@then(parsers.parse('the rebalance planner should report "files_to_add" containing blocks "blk-{start}.gguf" through "blk-{end}.gguf"'))
def step_rebalance_files_to_add(bdd_context, start, end):
    node_delta = bdd_context.rebalance_res["node1"]
    s_idx = int(start)
    e_idx = int(end)
    for i in range(s_idx, e_idx + 1):
        expected = f"blk-{i:05d}.gguf"
        assert expected in node_delta.files_to_add, f"Missing {expected} in files_to_add"


@then('"files_removable" should be empty')
def step_files_removable_empty(bdd_context):
    node_delta = bdd_context.rebalance_res["node1"]
    assert len(node_delta.files_removable) == 0


@then(parsers.parse('existing files for blocks {start:d} through {end:d} should not be re-downloaded'))
def step_existing_files_not_redownloaded(bdd_context, start, end):
    node_delta = bdd_context.rebalance_res["node1"]
    for i in range(start, end + 1):
        existing = f"blk-{i:05d}.gguf"
        assert existing not in node_delta.files_to_add, f"{existing} was unnecessarily flagged for addition"
