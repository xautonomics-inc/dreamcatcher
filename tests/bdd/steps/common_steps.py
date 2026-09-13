"""Common BDD step definitions shared across features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import requests
import pytest
from pytest_bdd import given, when, then, parsers

from tests.test_slicer import create_synthetic_gguf
import subprocess


def ensure_layer_library(lib_dir: Path, n_blocks: int = 4) -> None:
    """Ensure lib_dir has a valid sliced layer library."""
    if (lib_dir / "manifest.json").exists():
        return
    lib_dir.mkdir(parents=True, exist_ok=True)
    monolith = lib_dir.parent / "temp_monolith.gguf"
    create_synthetic_gguf(monolith, n_blocks=n_blocks)
    repo_root = Path(__file__).resolve().parents[3]
    slicer = repo_root / "examples" / "stage-runner" / "slice_gguf_layers.py"
    subprocess.run(["python3", str(slicer), str(monolith), str(lib_dir), "--force"], check=True)


@given(parsers.parse('a valid model library exists at "{lib_dir}"'))
def step_valid_model_library_exists(bdd_context, lib_dir):
    resolved = Path(bdd_context.resolve_placeholder(lib_dir))
    bdd_context.lib_dir = resolved
    ensure_layer_library(resolved, n_blocks=4)


@given(parsers.parse('a valid partitioned model library exists at "{lib_dir}"'))
def step_valid_partitioned_model_library_exists(bdd_context, lib_dir):
    resolved = Path(bdd_context.resolve_placeholder(lib_dir))
    bdd_context.lib_dir = resolved
    ensure_layer_library(resolved, n_blocks=4)


@given(parsers.parse('the library contains a valid "manifest.json" of format "{fmt}"'))
def step_library_contains_manifest(bdd_context, fmt):
    manifest_path = bdd_context.lib_dir / "manifest.json"
    assert manifest_path.exists(), f"manifest.json missing at {manifest_path}"
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data.get("format") == fmt


@given(parsers.parse('the library contains parts for all layers from 0 to "{total_layers}"'))
def step_library_contains_parts(bdd_context, total_layers):
    # If placeholder "<total_layers>", check at least some blk files exist
    if total_layers == "<total_layers>":
        blks = list(bdd_context.lib_dir.glob("blk-*.gguf"))
        assert len(blks) > 0
    else:
        n = int(total_layers)
        for i in range(n):
            assert (bdd_context.lib_dir / f"blk-{i:05d}.gguf").exists()


@given('the library contains mandatory stage parts "parts-embd.gguf" and "parts-output.gguf"')
def step_library_contains_mandatory_parts(bdd_context):
    assert (bdd_context.lib_dir / "parts-embd.gguf").exists()
    assert (bdd_context.lib_dir / "parts-output.gguf").exists()


@when(parsers.parse('I send a GET request to "{url}"'))
def step_send_get_request(bdd_context, url):
    resolved_url = bdd_context.resolve_placeholder(url)
    try:
        resp = requests.get(resolved_url, timeout=5.0)
        bdd_context.last_response = resp
    except Exception as exc:
        pytest.fail(f"HTTP GET request to {resolved_url} failed: {exc}")


@then(parsers.parse('the HTTP status code should be {status_code:d}'))
def step_http_status_code(bdd_context, status_code):
    assert bdd_context.last_response is not None, "No HTTP response recorded"
    assert bdd_context.last_response.status_code == status_code
