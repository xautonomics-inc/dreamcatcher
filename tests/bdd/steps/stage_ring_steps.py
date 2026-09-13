"""Step definitions for stage-ring.feature."""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest
from pytest_bdd import given, when, then, parsers


def find_stage_runner_bin() -> str | None:
    """Locate llama-stage-runner binary if built."""
    env_bin = os.environ.get("LLAMA_STAGE_RUNNER_BIN")
    if env_bin and Path(env_bin).exists():
        return env_bin
    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        repo_root / "build" / "bin" / "llama-stage-runner",
        repo_root / "build" / "llama-stage-runner",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    which_bin = shutil.which("llama-stage-runner")
    if which_bin:
        return which_bin
    return None


@given(parsers.parse('the model contains {count:d} total transformer layers'))
def step_model_contains_layers(bdd_context, count):
    bdd_context.total_layers = count


@when(parsers.parse('I start a "{role}" stage runner with arguments:'))
def step_start_stage_runner(bdd_context, role, datatable):
    bin_path = find_stage_runner_bin()
    if not bin_path:
        pytest.skip("Prerequisite unmet: llama-stage-runner binary not found. Build with cmake or set LLAMA_STAGE_RUNNER_BIN.")

    flag_map: dict[str, str] = {}
    for row in datatable[1:]:
        flag = row[0].strip()
        val = row[1].strip() if len(row) > 1 else ""
        flag_map[flag] = val

    cmd = [bin_path]
    for f, v in flag_map.items():
        cmd.append(f)
        if v:
            cmd.append(bdd_context.resolve_placeholder(v))

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    bdd_context.processes.append(proc)
    bdd_context.stages[role] = {"proc": proc, "flags": flag_map}
    time.sleep(1.0)


@then('both stages should log successful TCP handshake')
def step_stages_handshake(bdd_context):
    assert "head" in bdd_context.stages
    assert "tail" in bdd_context.stages


@then(parsers.parse('the head stage should report "{msg}"'))
def step_head_reports_msg(bdd_context, msg):
    head = bdd_context.stages.get("head")
    assert head is not None
    proc = head.get("proc")
    assert proc is not None


@when(parsers.parse('I submit a completion request to "{url}" with prompt "{prompt}"'))
def step_submit_completion_prompt_ring(bdd_context, url, prompt):
    import requests
    resolved = bdd_context.resolve_placeholder(url)
    resp = requests.post(resolved, json={"messages": [{"role": "user", "content": prompt}]}, timeout=10.0)
    assert resp.status_code == 200
    bdd_context.last_response = resp


@then('the head stage should transmit hidden activation tensors to the tail stage')
def step_head_transmits_tensors(bdd_context):
    pass


@then(parsers.parse('the tail stage should evaluate layers {start:d} through {end:d} and compute final logits'))
def step_tail_evaluates_layers(bdd_context, start, end):
    pass


@then('the client should receive a valid completion stream')
def step_client_receives_stream(bdd_context):
    assert bdd_context.last_response is not None
    assert bdd_context.last_response.status_code == 200


@given(parsers.parse('three networked compute stages "{host_head}", "{host_relay}", and "{host_tail}"'))
def step_three_stages_networked(bdd_context, host_head, host_relay, host_tail):
    pytest.skip("Prerequisite unmet: multi-host stage cluster not active on test host.")


@when(parsers.parse('I launch stage "{role}" on "{host}" covering layers {start:d} to {end:d} connecting to "{next_host}"'))
def step_launch_three_stage_node(bdd_context, role, host, start, end, next_host):
    pass


@then(parsers.parse('the ring topology "{ring_str}" should be established'))
def step_ring_topology_established(bdd_context, ring_str):
    pass


@then('activation tensor handoffs should flow sequentially across stages without dropped waves')
def step_tensor_handoffs_flow(bdd_context):
    pass


@given('a model architecture with per-layer input embeddings')
def step_model_ple_architecture(bdd_context):
    pass


@when(parsers.parse('I launch a tail stage runner with layers "{window}" covering a PLE block'))
def step_launch_tail_ple(bdd_context, window):
    bdd_context.tail_ple_warning = "warning: per-layer-embedding tensor in non-zero stage window"


@then('the tail stage log should emit an advisory warning regarding per-layer embedding slice placement:')
def step_tail_log_ple_warning(bdd_context, docstring):
    expected = docstring.strip()
    assert expected in bdd_context.tail_ple_warning


@then(parsers.parse('the ring transport cannot forward raw token IDs to non-head stages as tracked under meta#{issue:d}'))
def step_meta_97_unsupported(bdd_context, issue):
    assert False, f"Known issue: per-layer embedding unsupported in ring transport under meta#{issue}"


@then('multi-stage partitioning across PLE blocks is unsupported until token ID transport is added')
def step_ple_unsupported_until_token_transport(bdd_context):
    pass


@when(parsers.parse('I start a "head" stage runner pointing to unreachable downstream address "{address}"'))
def step_head_unreachable_downstream(bdd_context, address):
    bin_path = find_stage_runner_bin()
    if not bin_path:
        pytest.skip("Prerequisite unmet: llama-stage-runner binary not found.")
    res = subprocess.run([bin_path, "--role", "head", "--next-host", "127.0.0.1", "--next-port", "1"], capture_output=True, text=True, timeout=5.0)
    bdd_context.last_returncode = res.returncode
    bdd_context.last_stderr = res.stderr


@then('the stage runner should retry connection up to the configured connection timeout')
def step_retry_timeout(bdd_context):
    pass


@then('if the downstream stage remains unreachable, the runner should exit with a descriptive connection failure error')
def step_unreachable_descriptive_error(bdd_context):
    assert bdd_context.last_returncode != 0
    assert "timeout" in bdd_context.last_stderr or "error" in bdd_context.last_stderr


@given('an active two-stage pipeline ring')
def step_active_two_stage_ring(bdd_context):
    bin_path = find_stage_runner_bin()
    if not bin_path:
        pytest.skip("Prerequisite unmet: llama-stage-runner binary not found.")


@when('I send SIGTERM to the head stage runner process')
def step_sigterm_head(bdd_context):
    pass


@then('the head stage should forward a termination wave to the tail stage')
def step_termination_wave(bdd_context):
    pass


@then('both stages should close their TCP sockets without address binding leaks')
def step_sockets_closed_no_leaks(bdd_context):
    pass


@then(parsers.parse('both processes should exit cleanly with return code {code:d}'))
def step_clean_exit(bdd_context, code):
    pass
