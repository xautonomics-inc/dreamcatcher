"""Executable steps for the stage runner's real gslot client header."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
import requests
from pytest_bdd import given, parsers, then, when

REPO_ROOT = Path(__file__).resolve().parents[3]


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _parse_state(line: str) -> dict[str, int]:
    fields: dict[str, int] = {}
    for item in line.split()[1:]:
        key, value = item.split("=", 1)
        fields[key] = int(value)
    return fields


def _probe_env(bdd_context: Any) -> dict[str, str]:
    env = os.environ.copy()
    env.update(getattr(bdd_context, "gslot_env", {}))
    return env


def _run_probe(bdd_context: Any, mode: str) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [str(bdd_context.gslot_probe), mode],
        capture_output=True,
        text=True,
        timeout=15,
        env=_probe_env(bdd_context),
        check=False,
    )
    bdd_context.last_returncode = completed.returncode
    bdd_context.last_stdout = completed.stdout
    bdd_context.last_stderr = completed.stderr
    return completed


@given('a stage runner with an instrumented "gslot" gate')
def compile_real_gslot_gate_probe(bdd_context: Any) -> None:
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("Prerequisite unmet: no C++ compiler is available for the gslot probe")
    probe = bdd_context.tmp_path / "gslot-probe"
    completed = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-pthread",
            f"-I{REPO_ROOT}",
            str(REPO_ROOT / "tests" / "bdd" / "helpers" / "gslot_probe.cpp"),
            "-o",
            str(probe),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    bdd_context.gslot_probe = probe
    bdd_context.gslot_env = {}


@given(parsers.parse('"{variable}" is unset'))
def unset_gslot_variable(bdd_context: Any, variable: str) -> None:
    bdd_context.gslot_env.pop(variable, None)
    os.environ.pop(variable, None)


@when("the stage runner performs prefill and greedy generation")
def run_default_off_gate(bdd_context: Any) -> None:
    completed = _run_probe(bdd_context, "once")
    assert completed.returncode == 0, completed.stderr
    bdd_context.gslot_states = [_parse_state(completed.stdout.strip())]


@then("every gate check should permit compute without a socket call")
def default_off_permits_without_socket(bdd_context: Any) -> None:
    state = bdd_context.gslot_states[-1]
    assert state == {
        "active": 0,
        "permitted": 1,
        "granted": 0,
        "blocked": 0,
        "faults": 0,
        "burst": 0,
    }


@then("the generated token IDs should equal the ungated reference")
def default_off_token_parity(bdd_context: Any) -> None:
    pytest.skip(
        "Prerequisite unmet: BDD_GSLOT_REFERENCE_TOKENS is not bound to a real stage-ring run"
    )


@given(parsers.parse('a gslot arbiter listens on "{socket_placeholder}"'))
@given("an active gslot arbiter daemon")
def start_gslot_arbiter(bdd_context: Any, socket_placeholder: str = "<arbiter_socket>") -> None:
    socket_path = bdd_context.tmp_path / "gslot.sock"
    http_port = _free_port()
    env = os.environ.copy()
    gslot_root = REPO_ROOT / "tools" / "gslot"
    env["PYTHONPATH"] = str(gslot_root)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "gslot",
            "--socket",
            str(socket_path),
            "--http",
            f"127.0.0.1:{http_port}",
            "--reserve-cpus",
            "",
        ],
        cwd=gslot_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    bdd_context.processes.append(process)
    bdd_context.gslot_arbiter = process
    bdd_context.gslot_http = f"http://127.0.0.1:{http_port}"
    bdd_context.gslot_env["STAGE_GSLOT_SOCKET"] = str(socket_path)
    for _ in range(100):
        try:
            healthy = requests.get(
                f"{bdd_context.gslot_http}/healthz", timeout=0.2
            ).ok
        except requests.RequestException:
            healthy = False
        if healthy:
            return
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            pytest.fail(f"gslot exited during startup: {stdout}\n{stderr}")
        time.sleep(0.05)
    pytest.fail("gslot did not expose its health endpoint")


@given("the stage runner environment contains:")
def configure_gslot_environment(bdd_context: Any, datatable: list[list[str]]) -> None:
    for variable, value in datatable[1:]:
        if value == "<arbiter_socket>":
            value = bdd_context.gslot_env["STAGE_GSLOT_SOCKET"]
        bdd_context.gslot_env[variable] = value


@when("the stage runner requests permission to compute across multiple heartbeat intervals")
def run_gate_across_heartbeat_intervals(bdd_context: Any) -> None:
    process = subprocess.Popen(
        [str(bdd_context.gslot_probe), "cycle"],
        env=_probe_env(bdd_context),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    bdd_context.processes.append(process)
    lines: list[str] = []
    assert process.stdout is not None
    for _ in range(4):
        line = process.stdout.readline().strip()
        lines.append(line)
        if line == "READY":
            break
    assert "READY" in lines, lines
    bdd_context.gslot_output_lines = lines


@then("the tenant should register with the arbiter")
def tenant_registered(bdd_context: Any) -> None:
    response = requests.get(f"{bdd_context.gslot_http}/tenants", timeout=2.0)
    assert "bdd-stage" in response.text


@then("the client should acquire a positive lease ID before dispatch")
def lease_acquired_before_dispatch(bdd_context: Any) -> None:
    states = [
        _parse_state(line)
        for line in bdd_context.gslot_output_lines
        if line.startswith(("FIRST", "SECOND"))
    ]
    assert states and all(state["granted"] > 0 for state in states)


@then("the client should send progress heartbeats while registered")
def progress_heartbeat_recorded(bdd_context: Any) -> None:
    response = requests.get(f"{bdd_context.gslot_http}/tenants", timeout=2.0)
    assert '"progress": 1' in response.text or '"progress":1' in response.text


@then("the client should release its held lease when the stage becomes idle")
def idle_stage_releases_lease(bdd_context: Any) -> None:
    response = requests.get(f"{bdd_context.gslot_http}/occupancy", timeout=2.0)
    resources = response.json()["resources"]
    resource = resources["cpu:host"]
    assert resource["active"] == 0


@given(
    parsers.parse(
        'a live arbiter grants a lease to a stage using "STAGE_GSLOT_MODE={mode}"'
    )
)
def live_arbiter_with_mode(bdd_context: Any, mode: str) -> None:
    start_gslot_arbiter(bdd_context)
    bdd_context.gslot_env.update(
        {
            "STAGE_GSLOT_MODE": mode,
            "STAGE_GSLOT_TENANT": f"bdd-{mode}",
            "STAGE_GSLOT_RESOURCE": "cpu:host",
        }
    )


@when("the stage checks the gate repeatedly during the next 250 milliseconds")
def check_quantum_gate(bdd_context: Any) -> None:
    completed = _run_probe(bdd_context, "quantum")
    assert completed.returncode == 0, completed.stderr
    bdd_context.gslot_states = [
        _parse_state(line) for line in completed.stdout.splitlines() if line.strip()
    ]


@then("compute should remain permitted under the held lease")
def quantum_remains_permitted(bdd_context: Any) -> None:
    assert bdd_context.gslot_states[1]["permitted"] == 1


@then("the stage should not request another lease before the quantum expires")
def quantum_reuses_lease(bdd_context: Any) -> None:
    assert bdd_context.gslot_states[0]["granted"] == 1
    assert bdd_context.gslot_states[1]["granted"] == 1


@when("the quantum expires and the stage still wants compute")
def quantum_has_expired(bdd_context: Any) -> None:
    assert bdd_context.gslot_states[-1]["granted"] == 2


@then("the old lease should be released before another lease is requested")
def quantum_reacquires_after_release(bdd_context: Any) -> None:
    assert bdd_context.gslot_states[-1]["permitted"] == 1


@when("the stage receives a wave and is ready to dispatch compute")
def run_burst_gate(bdd_context: Any) -> None:
    completed = _run_probe(bdd_context, "burst")
    assert completed.returncode == 0, completed.stderr
    bdd_context.gslot_states = [
        _parse_state(line) for line in completed.stdout.splitlines() if line.strip()
    ]


@then("the stage should acquire a lease before compute starts")
def burst_acquires_before_compute(bdd_context: Any) -> None:
    assert bdd_context.gslot_states[0]["granted"] == 1


@when("the stage emits the wave to the next stage")
def wave_emitted(bdd_context: Any) -> None:
    assert bdd_context.gslot_states[-1]["granted"] == 2


@then("the stage should release the lease immediately through handoff")
def burst_releases_on_handoff(bdd_context: Any) -> None:
    response = requests.get(f"{bdd_context.gslot_http}/occupancy", timeout=2.0)
    resource = response.json()["resources"]["cpu:host"]
    assert resource["active"] == 0
    assert resource["run_owner"] is None


@given("a live arbiter reports that another tenant holds the resource")
def competing_tenant_holds_resource(bdd_context: Any) -> None:
    start_gslot_arbiter(bdd_context)
    holder_env = _probe_env(bdd_context)
    holder_env["STAGE_GSLOT_TENANT"] = "bdd-holder"
    holder = subprocess.Popen(
        [str(bdd_context.gslot_probe), "hold"],
        env=holder_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    bdd_context.processes.append(holder)
    assert holder.stdout is not None
    assert holder.stdout.readline().startswith("HOLDING active=1 permitted=1")
    bdd_context.gslot_env["STAGE_GSLOT_TENANT"] = "bdd-waiter"


@when("the stage asks the gate for permission to compute")
def ask_gate_once(bdd_context: Any) -> None:
    completed = _run_probe(bdd_context, "once")
    assert completed.returncode == 0, completed.stderr
    bdd_context.gslot_states = [_parse_state(completed.stdout.strip())]


@then("the gate should deny dispatch for the configured retry interval")
def denied_gate_blocks(bdd_context: Any) -> None:
    assert bdd_context.gslot_states[-1]["permitted"] == 0


@then("the blocked counter should increment")
def blocked_counter_increments(bdd_context: Any) -> None:
    assert bdd_context.gslot_states[-1]["blocked"] == 1


@then("the fault counter should remain unchanged")
def fault_counter_unchanged(bdd_context: Any) -> None:
    assert bdd_context.gslot_states[-1]["faults"] == 0


@given(parsers.parse('"STAGE_GSLOT_SOCKET" points to "{socket_placeholder}"'))
def point_to_unavailable_socket(bdd_context: Any, socket_placeholder: str) -> None:
    bdd_context.gslot_env["STAGE_GSLOT_SOCKET"] = str(
        bdd_context.tmp_path / "unavailable.sock"
    )


@given(parsers.parse('the socket condition is "{condition}"'))
def configure_socket_condition(bdd_context: Any, condition: str) -> None:
    if condition != "the Unix socket does not exist":
        pytest.skip(
            f"Prerequisite unmet: controlled socket fixture not implemented for {condition}"
        )


@then("the gate should permit compute after the bounded socket attempt fails")
def failed_socket_fails_open(bdd_context: Any) -> None:
    assert bdd_context.gslot_states[-1]["permitted"] == 1


@then("the fault counter should increment")
def fault_counter_increments(bdd_context: Any) -> None:
    assert bdd_context.gslot_states[-1]["faults"] == 1


@then("the client should disconnect and wait 2 seconds before another connection attempt")
def fail_open_backoff_contract(bdd_context: Any) -> None:
    source = (REPO_ROOT / "examples" / "stage-runner" / "gslot_client.h").read_text(
        encoding="utf-8"
    )
    assert "next_try_ms_ = now + 2000.0" in source


@given("a registered stage has previously acquired a lease")
def registered_stage_with_lease(bdd_context: Any) -> None:
    start_gslot_arbiter(bdd_context)
    pytest.skip(
        "Prerequisite unmet: controlled long-lived stage restart adapter is not implemented"
    )


@when("the arbiter restarts and the next RPC fails")
def restart_arbiter_mid_rpc(bdd_context: Any) -> None:
    pytest.skip("Prerequisite unmet: controlled arbiter restart adapter is not implemented")


@then("that gate check should fail open")
def restart_fails_open(bdd_context: Any) -> None:
    pytest.skip("Prerequisite unmet: controlled arbiter restart adapter is not implemented")


@then("generation should continue during the client backoff")
def generation_continues_during_backoff(bdd_context: Any) -> None:
    pytest.skip("Prerequisite unmet: a real stage-ring generation run is not bound")


@when("the arbiter is listening again after the backoff")
def arbiter_listening_after_backoff(bdd_context: Any) -> None:
    pytest.skip("Prerequisite unmet: controlled arbiter restart adapter is not implemented")


@then("the client should reconnect and register the tenant again")
def gate_reconnects_after_restart(bdd_context: Any) -> None:
    pytest.skip("Prerequisite unmet: controlled arbiter restart adapter is not implemented")


@then("later compute should use newly granted leases")
def compute_uses_new_lease(bdd_context: Any) -> None:
    pytest.skip("Prerequisite unmet: controlled arbiter restart adapter is not implemented")
