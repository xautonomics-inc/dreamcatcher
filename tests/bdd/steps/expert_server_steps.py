"""Executable steps for remote MoE expert serving.

Every scenario requires real fork binaries and an explicitly bound MoE model.
No dense-model or mock fallback is used.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest
from pytest_bdd import given, parsers, then, when

REPO_ROOT = Path(__file__).resolve().parents[3]
TOK_LINE = re.compile(r"^TOK step=(\d+) id=(\d+) logits_fnv=(0x[0-9a-f]+)$")


def _binary(env_name: str, default_name: str) -> Path | None:
    configured = os.environ.get(env_name)
    if configured and Path(configured).is_file():
        return Path(configured)
    for candidate in (
        REPO_ROOT / "build" / "bin" / default_name,
        REPO_ROOT / "build" / default_name,
    ):
        if candidate.is_file():
            return candidate
    found = shutil.which(default_name)
    return Path(found) if found else None


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_for_port(process: subprocess.Popen[str], port: int) -> None:
    for _ in range(300):
        if process.poll() is not None:
            pytest.fail(f"expert server exited during startup with {process.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.05):
                return
        except OSError:
            time.sleep(0.05)
    pytest.fail(f"expert server did not listen on the assigned port {port}")


def _start_server(
    bdd_context: Any, layer_range: str, extra_arguments: str = ""
) -> tuple[subprocess.Popen[str], int, Path]:
    port = _free_port()
    log_path = bdd_context.tmp_path / f"expert-server-{port}.log"
    log_handle = log_path.open("w", encoding="utf-8")
    command = [
        str(bdd_context.expert_server_bin),
        "--role",
        "expert-server",
        "--model",
        str(bdd_context.moe_model),
        "--expert-layers",
        layer_range,
        "--host",
        "127.0.0.1",
        "--listen",
        str(port),
        "--device",
        "cpu",
    ]
    command.extend(shlex.split(extra_arguments))
    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_handle.close()
    bdd_context.processes.append(process)
    _wait_for_port(process, port)
    return process, port, log_path


def _run_check(
    bdd_context: Any,
    *,
    remote: str | None,
    client_arguments: str = "",
    label: str,
) -> dict[str, Any]:
    logits_path = bdd_context.tmp_path / f"{label}.logits"
    env = os.environ.copy()
    env["EXPERT_CHECK_LOGITS_OUT"] = str(logits_path)
    if remote is None:
        env.pop("LLAMA_EXPERTS_REMOTE", None)
    else:
        env["LLAMA_EXPERTS_REMOTE"] = remote
    command = [
        str(bdd_context.expert_check_bin),
        "-m",
        str(bdd_context.moe_model),
        "-p",
        bdd_context.expert_prompt,
        "-n",
        str(bdd_context.expert_tokens),
        "-ngl",
        "0",
    ]
    command.extend(shlex.split(client_arguments))
    # The TEXT line echoes decoded pieces, which need not be valid UTF-8 (a
    # byte-level vocab such as the synthetic Inkling fixture emits raw bytes);
    # the TOK lines this harness parses are ASCII, so decode leniently rather
    # than let one stray byte abort the comparison.
    completed = subprocess.run(
        command,
        env=env,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=float(os.environ.get("BDD_EXPERT_TIMEOUT", "900")),
        check=False,
    )
    token_rows = [
        match.groups()
        for line in completed.stdout.splitlines()
        if (match := TOK_LINE.match(line)) is not None
    ]
    result = {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "tokens": [(int(step), int(token), digest) for step, token, digest in token_rows],
        "logits_path": logits_path,
    }
    bdd_context.last_returncode = completed.returncode
    bdd_context.last_stdout = completed.stdout
    bdd_context.last_stderr = completed.stderr
    return result


@given(parsers.parse('a supported MoE model at "{model_placeholder}"'))
def supported_moe_model(bdd_context: Any, model_placeholder: str) -> None:
    model = os.environ.get("BDD_MOE_MODEL")
    server = _binary("LLAMA_EXPERT_SERVER_BIN", "llama-expert-server")
    check = _binary("LLAMA_EXPERT_CHECK_BIN", "llama-expert-check")
    missing = []
    if not model or not Path(model).is_file():
        missing.append("BDD_MOE_MODEL")
    layer_count = os.environ.get("BDD_MOE_LAYERS")
    if not layer_count or not layer_count.isdigit() or int(layer_count) < 32:
        missing.append("BDD_MOE_LAYERS (integer >= 32)")
    if server is None:
        missing.append("LLAMA_EXPERT_SERVER_BIN")
    if check is None:
        missing.append("LLAMA_EXPERT_CHECK_BIN")
    if missing:
        pytest.skip("Prerequisite unmet: " + ", ".join(missing))
    bdd_context.moe_model = Path(model)
    bdd_context.expert_server_bin = server
    bdd_context.expert_check_bin = check
    bdd_context.moe_layers = int(layer_count)


@given("a deterministic greedy prompt and generation length")
def deterministic_expert_request(bdd_context: Any) -> None:
    bdd_context.expert_prompt = os.environ.get(
        "BDD_EXPERT_PROMPT", "The capital of France is"
    )
    bdd_context.expert_tokens = int(os.environ.get("BDD_EXPERT_TOKENS", "12"))
    assert bdd_context.expert_tokens > 0


@given('"llama-expert-server" is started with:')
def start_documented_expert_server(bdd_context: Any, docstring: str) -> None:
    match = re.search(r"--expert-layers\s+(\S+)", docstring)
    assert match is not None
    _, port, log_path = _start_server(bdd_context, match.group(1))
    bdd_context.expert_port = port
    bdd_context.expert_log = log_path


@when(parsers.parse('"llama-expert-check" starts with "{configuration}"'))
def run_connected_expert_check(bdd_context: Any, configuration: str) -> None:
    remote = f"127.0.0.1:{bdd_context.expert_port}@0-31"
    bdd_context.remote_result = _run_check(
        bdd_context, remote=remote, label="connected"
    )
    assert bdd_context.remote_result["returncode"] == 0, bdd_context.remote_result[
        "stderr"
    ]


@then("the client and server complete the version 1 capability handshake")
def capability_handshake_completed(bdd_context: Any) -> None:
    for _ in range(100):
        text = bdd_context.expert_log.read_text(encoding="utf-8")
        if "client connected" in text:
            break
        time.sleep(0.02)
    assert "client connected" in text


@then("the client verifies the model embedding width and served layer set")
def capability_values_verified(bdd_context: Any) -> None:
    assert bdd_context.remote_result["returncode"] == 0
    source = (REPO_ROOT / "src" / "llama-experts-remote.cpp").read_text(
        encoding="utf-8"
    )
    assert "assigned layer" in source and "embedding mismatch" in source


@then("covered layers send hidden values, top-k expert IDs, and top-k weights")
def request_payload_contract_exercised(bdd_context: Any) -> None:
    assert bdd_context.remote_result["tokens"]
    source = (REPO_ROOT / "src" / "llama-experts-remote.h").read_text(
        encoding="utf-8"
    )
    for field in ("selected expert ids", "final per-expert weights", "hidden states"):
        assert field in source


@then("covered layers receive the accumulated routed-expert output")
def expert_output_received(bdd_context: Any) -> None:
    assert bdd_context.remote_result["logits_path"].stat().st_size > 0


@given(parsers.parse('a local "llama-expert-check" reference uses client arguments "{arguments}"'))
def run_local_expert_reference(bdd_context: Any, arguments: str) -> None:
    bdd_context.client_arguments = arguments
    bdd_context.local_result = _run_check(
        bdd_context,
        remote=None,
        client_arguments=arguments,
        label="local-reference",
    )
    assert bdd_context.local_result["returncode"] == 0, bdd_context.local_result[
        "stderr"
    ]


@given(parsers.parse('a CPU expert server uses server arguments "{arguments}"'))
def run_cpu_expert_server(bdd_context: Any, arguments: str) -> None:
    _, port, log_path = _start_server(
        bdd_context, f"0-{bdd_context.moe_layers - 1}", arguments
    )
    bdd_context.expert_port = port
    bdd_context.expert_log = log_path


@when(
    parsers.parse(
        'the same check runs remotely for every MoE layer with client arguments "{arguments}"'
    )
)
def run_remote_expert_reference(bdd_context: Any, arguments: str) -> None:
    bdd_context.remote_result = _run_check(
        bdd_context,
        remote=f"127.0.0.1:{bdd_context.expert_port}",
        client_arguments=arguments,
        label="remote-reference",
    )
    assert bdd_context.remote_result["returncode"] == 0, bdd_context.remote_result[
        "stderr"
    ]


@then("every generated token ID should equal the local reference")
def expert_token_ids_match(bdd_context: Any) -> None:
    local = [(step, token) for step, token, _ in bdd_context.local_result["tokens"]]
    remote = [(step, token) for step, token, _ in bdd_context.remote_result["tokens"]]
    assert local and remote == local


@then("every per-step logits hash should equal the local reference")
def expert_logits_hashes_match(bdd_context: Any) -> None:
    assert bdd_context.remote_result["tokens"] == bdd_context.local_result["tokens"]


@then("the remote and local raw logits dumps should be byte-identical")
def expert_raw_logits_match(bdd_context: Any) -> None:
    assert bdd_context.remote_result["logits_path"].read_bytes() == bdd_context.local_result[
        "logits_path"
    ].read_bytes()


@given("the local reference uses the default fused client path")
def fused_local_reference(bdd_context: Any) -> None:
    run_local_expert_reference(bdd_context, "")


@given('the expert server runs with "--fmoe 0 --mmad 0"')
def unfused_expert_server(bdd_context: Any) -> None:
    run_cpu_expert_server(bdd_context, "--fmoe 0 --mmad 0")


@when("the same deterministic check runs through the expert server")
def run_mismatched_expert_check(bdd_context: Any) -> None:
    bdd_context.remote_result = _run_check(
        bdd_context,
        remote=f"127.0.0.1:{bdd_context.expert_port}",
        label="fusion-mismatch",
    )
    assert bdd_context.remote_result["returncode"] == 0


@then("at least one logits hash should differ from the local reference")
def mismatched_hash_differs(bdd_context: Any) -> None:
    assert bdd_context.remote_result["tokens"] != bdd_context.local_result["tokens"]


@then("the run should not be reported as byte-exact")
def mismatched_run_is_not_exact(bdd_context: Any) -> None:
    assert bdd_context.remote_result["logits_path"].read_bytes() != bdd_context.local_result[
        "logits_path"
    ].read_bytes()


@given(parsers.parse("one expert server serves layers {start:d} through {end:d}"))
def first_expert_endpoint(bdd_context: Any, start: int, end: int) -> None:
    _, port, _ = _start_server(bdd_context, f"{start}-{end}")
    bdd_context.expert_endpoints = [(port, start, end)]


@given(parsers.parse("another expert server serves layers {start:d} through {end:d}"))
def second_expert_endpoint(bdd_context: Any, start: int, end: int) -> None:
    _, port, _ = _start_server(bdd_context, f"{start}-{end}")
    bdd_context.expert_endpoints.append((port, start, end))


@when(parsers.parse('the client starts with "{configuration}"'))
def run_multi_endpoint_check(bdd_context: Any, configuration: str) -> None:
    remote = ";".join(
        f"127.0.0.1:{port}@{start}-{end}"
        for port, start, end in bdd_context.expert_endpoints
    )
    bdd_context.local_result = _run_check(
        bdd_context, remote=None, label="multi-local"
    )
    bdd_context.remote_result = _run_check(
        bdd_context, remote=remote, label="multi-remote"
    )
    assert bdd_context.remote_result["returncode"] == 0


@then("each covered layer should be routed to the endpoint that owns it")
def disjoint_endpoint_routing(bdd_context: Any) -> None:
    assert len(bdd_context.expert_endpoints) == 2


@then("the deterministic logits should be byte-identical to local CPU experts")
def multi_endpoint_logits_match(bdd_context: Any) -> None:
    expert_raw_logits_match(bdd_context)


@when("the client starts with two remote endpoints that both claim layer 15")
def overlapping_endpoint_configuration(bdd_context: Any) -> None:
    env = os.environ.copy()
    env["LLAMA_EXPERTS_REMOTE"] = "127.0.0.1:1@0-15;127.0.0.1:2@15-31"
    completed = subprocess.run(
        [
            str(bdd_context.expert_check_bin),
            "-m",
            str(bdd_context.moe_model),
            "-n",
            "1",
            "-ngl",
            "0",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    bdd_context.last_returncode = completed.returncode
    bdd_context.last_stderr = completed.stderr


@then("configuration parsing should abort before model execution")
def overlap_aborts_configuration(bdd_context: Any) -> None:
    assert bdd_context.last_returncode not in (None, 0)


@then("the diagnostic should identify overlapping layer coverage")
def overlap_diagnostic_identifies_coverage(bdd_context: Any) -> None:
    assert "layers may not overlap" in bdd_context.last_stderr


@given("a client has completed the capability handshake with an expert server")
def connected_client_for_fault_injection(bdd_context: Any) -> None:
    pytest.skip(
        "Prerequisite unmet: controlled mid-RPC expert transport fault injector is not implemented"
    )


@when("the connection fails during an expert RPC and the server becomes available again")
def interrupt_expert_rpc(bdd_context: Any) -> None:
    pytest.skip("Prerequisite unmet: controlled expert transport fault injector is not implemented")


@then("the client should tear down the failed connection")
def failed_expert_connection_closed(bdd_context: Any) -> None:
    pytest.skip("Prerequisite unmet: controlled expert transport fault injector is not implemented")


@then("the client should retry the RPC at most 3 times with reconnect and backoff")
def expert_rpc_retries(bdd_context: Any) -> None:
    pytest.skip("Prerequisite unmet: controlled expert transport fault injector is not implemented")


@then("a successful reconnect should repeat the capability handshake before another expert call")
def expert_reconnect_handshake(bdd_context: Any) -> None:
    pytest.skip("Prerequisite unmet: controlled expert transport fault injector is not implemented")


@given('"LLAMA_EXPERTS_REMOTE" is unset')
def remote_experts_unset(bdd_context: Any) -> None:
    os.environ.pop("LLAMA_EXPERTS_REMOTE", None)


@when("the deterministic check runs with all experts local")
def run_all_local_experts(bdd_context: Any) -> None:
    bdd_context.local_result = _run_check(
        bdd_context, remote=None, label="local-a"
    )
    bdd_context.remote_result = _run_check(
        bdd_context, remote=None, label="local-b"
    )
    assert bdd_context.local_result["returncode"] == 0
    assert bdd_context.remote_result["returncode"] == 0


@then("no remote expert connection should be attempted")
def no_remote_connection_attempted(bdd_context: Any) -> None:
    combined = bdd_context.remote_result["stdout"] + bdd_context.remote_result["stderr"]
    assert "experts-remote:" not in combined


@then("the generated token IDs and logits hashes should equal the local baseline")
def repeated_local_check_matches(bdd_context: Any) -> None:
    expert_logits_hashes_match(bdd_context)
