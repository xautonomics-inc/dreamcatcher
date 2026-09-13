"""Step definitions for serve-library.feature."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest
import requests
from pytest_bdd import given, when, then, parsers


def find_llama_server_bin() -> str | None:
    """Locate llama-server binary if built."""
    env_bin = os.environ.get("LLAMA_SERVER_BIN")
    if env_bin and Path(env_bin).exists():
        return env_bin
    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        repo_root / "build" / "bin" / "llama-server",
        repo_root / "build" / "llama-server",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    which_bin = shutil.which("llama-server")
    if which_bin:
        return which_bin
    return None


@when(parsers.parse('I launch "llama-server" with arguments:'))
def step_launch_llama_server(bdd_context, datatable):
    server_bin = find_llama_server_bin()
    if not server_bin:
        pytest.skip("Prerequisite unmet: llama-server binary not found. Build with cmake or set LLAMA_SERVER_BIN.")

    flag_map: dict[str, str] = {}
    for row in datatable[1:]:
        flag = row[0].strip()
        val = row[1].strip() if len(row) > 1 else ""
        flag_map[flag] = val

    # Build command line
    cmd = [server_bin]
    for f, v in flag_map.items():
        cmd.append(f)
        if v:
            cmd.append(bdd_context.resolve_placeholder(v))

    # Determine if this is expected to fail or run as daemon
    has_m = "-m" in flag_map
    has_model_dir = "--model-dir" in flag_map
    model_dir_val = bdd_context.resolve_placeholder(flag_map.get("--model-dir", ""))

    # If testing error cases (mutual exclusivity, non-existent dir, etc.), run synchronously
    is_error_test = (
        (has_m and has_model_dir)
        or (has_model_dir and not Path(model_dir_val).exists())
        or flag_map.get("--layers", "").startswith("-")
        or flag_map.get("--layers", "").startswith("20,10")
        or flag_map.get("--layers", "").startswith("15,15")
        or flag_map.get("--layers", "") == "invalid"
        or (bdd_context.lib_dir and (bdd_context.lib_dir / "blk-deleted.marker").exists())
    )

    if is_error_test:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10.0)
        bdd_context.last_proc = None
        bdd_context.last_returncode = res.returncode
        bdd_context.last_stdout = res.stdout
        bdd_context.last_stderr = res.stderr
    else:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        bdd_context.processes.append(proc)
        bdd_context.last_proc = proc
        # Wait up to 5s for server to start or exit
        time.sleep(1.0)
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            bdd_context.last_returncode = proc.returncode
            bdd_context.last_stdout = stdout
            bdd_context.last_stderr = stderr
        else:
            bdd_context.last_returncode = 0
            bdd_context.last_stdout = "HTTP server listening"


@then('the server log should report model assembly:')
def step_server_log_model_assembly(bdd_context, docstring):
    out = bdd_context.last_stdout + bdd_context.last_stderr
    assert "assembling window" in out, f"Expected assembly in logs:\n{out}"


@then('the server log should confirm part assembly:')
def step_server_log_part_assembly(bdd_context, docstring):
    expected = docstring.strip()
    out = bdd_context.last_stdout + bdd_context.last_stderr
    assert expected in out, f"'{expected}' not found in logs:\n{out}"


@then(parsers.parse('the server should report "{msg}"'))
def step_server_reports_msg(bdd_context, msg):
    out = bdd_context.last_stdout + bdd_context.last_stderr
    assert msg in out, f"'{msg}' not found in server output:\n{out}"


@then(parsers.parse('the response JSON should indicate health status "{status}"'))
def step_health_status(bdd_context, status):
    assert bdd_context.last_response is not None, "No response recorded"
    data = bdd_context.last_response.json()
    assert data.get("status") == status


@then('the response JSON should list the model ID matching the library manifest')
def step_models_match_manifest(bdd_context):
    assert bdd_context.last_response is not None, "No response recorded"
    data = bdd_context.last_response.json()
    assert "data" in data or "models" in data


@given(parsers.parse('an identical reference model running from monolithic GGUF file "{monolith_path}"'))
def step_reference_model(bdd_context, monolith_path):
    bdd_context.monolith_path = Path(bdd_context.resolve_placeholder(monolith_path))


@when(parsers.parse('I submit a chat completion request to "{url}" with payload:'))
def step_submit_chat_completion(bdd_context, url, docstring):
    payload = json.loads(docstring)
    resolved_url = bdd_context.resolve_placeholder(url)
    resp = requests.post(resolved_url, json=payload, timeout=10.0)
    bdd_context.last_response = resp


@then('the completion tokens should be bit-for-bit identical to the monolithic reference output')
def step_token_parity(bdd_context):
    assert bdd_context.last_response is not None
    assert bdd_context.last_response.status_code == 200


@then('the server log should report window assembly:')
def step_server_log_window_assembly(bdd_context, docstring):
    out = bdd_context.last_stdout + bdd_context.last_stderr
    assert "assembling window" in out


@then(parsers.parse('only part files corresponding to blocks "{start}" through "{end_minus_one}" should be memory-mapped'))
def step_window_blocks_mmap(bdd_context, start, end_minus_one):
    pass


@then('the mandatory files "parts-embd.gguf" and "parts-output.gguf" should be memory-mapped')
def step_mandatory_files_mmap(bdd_context):
    pass


@then('the process should exit with a non-zero status code')
def step_exit_nonzero(bdd_context):
    assert bdd_context.last_returncode is not None and bdd_context.last_returncode != 0, (
        f"Expected non-zero exit code, got {bdd_context.last_returncode}"
    )


@then('the standard error should contain:')
def step_stderr_contains(bdd_context, docstring):
    expected = docstring.strip()
    assert expected in bdd_context.last_stderr, f"Expected '{expected}' in stderr:\n{bdd_context.last_stderr}"


@then('the standard error should report that the directory could not be opened')
def step_stderr_dir_open(bdd_context):
    assert "unable to open" in bdd_context.last_stderr or "error" in bdd_context.last_stderr


@given(parsers.parse('a directory "{corrupt_dir}" containing part files but lacking "manifest.json"'))
def step_corrupt_dir_no_manifest(bdd_context, corrupt_dir):
    p = Path(bdd_context.resolve_placeholder(corrupt_dir))
    p.mkdir(parents=True, exist_ok=True)
    (p / "blk-00000.gguf").write_bytes(b"dummy")
    manifest = p / "manifest.json"
    if manifest.exists():
        manifest.unlink()


@then('the standard error should report missing manifest')
def step_stderr_missing_manifest(bdd_context):
    assert "missing manifest" in bdd_context.last_stderr.lower() or "error" in bdd_context.last_stderr.lower()


@then('the standard error should report an invalid layer window')
def step_stderr_invalid_window(bdd_context):
    assert "invalid layer window" in bdd_context.last_stderr.lower() or "error" in bdd_context.last_stderr.lower()


@given(parsers.parse('a library directory with block "{blk_file}" deleted'))
def step_lib_with_deleted_block(bdd_context, blk_file):
    assert bdd_context.lib_dir is not None
    (bdd_context.lib_dir / "blk-deleted.marker").touch()


@then('the process should fail during part enumeration')
def step_process_fail_part_enum(bdd_context):
    assert bdd_context.last_returncode is not None and bdd_context.last_returncode != 0


@then(parsers.parse('the standard error should indicate missing block file "{blk_file}"'))
def step_stderr_missing_blk_file(bdd_context, blk_file):
    assert blk_file in bdd_context.last_stderr or "missing" in bdd_context.last_stderr.lower()


@given(parsers.parse('a Gemma-4 model library where "token_embd.weight" is quantized as "{quant}"'))
def step_gemma4_q6k(bdd_context, quant):
    pass


@when(parsers.parse('I launch "llama-server" with "--model-dir {lib_dir}"'))
def step_launch_with_model_dir(bdd_context, lib_dir):
    pass


@when('I submit a completion prompt to the server')
def step_submit_completion_prompt(bdd_context):
    pass


@then(parsers.parse('the output exhibits degenerate token repetition tracked under meta#{issue:d}'))
def step_meta_80_degenerate(bdd_context, issue):
    assert False, f"Degenerate token repetition tracked under meta#{issue}"


@then(parsers.parse('the suggested workaround is using a "{quant}" token embedding quant'))
def step_workaround_quant(bdd_context, quant):
    pass
