"""BDD steps for the @d4c @remote-experts Inkling scenario (features/inkling.feature).

Lane D4c: Inkling's routed experts served by the remote expert-server
(examples/stage-runner/expert-server.cpp --moe-form inkling) are byte-exact
against running them in-process (src/graphs/build_inkling.cpp).

Comparand: this build's OWN in-process run — same binary, same host, same
CPU — not the D0 oracle. D4c splits one graph across two processes within
one lineage, so the gate is byte-identity, the definition the expert-server
feature already pins (steps/expert_server_steps.py): every decoded token id,
every per-step FNV-1a logits hash and the raw logits dump must be equal.
That module's helpers (_binary, _start_server, _run_check) are reused
unchanged so both features prove the same thing the same way.

Model: by default the tiny synthetic Inkling fixture generated IN-STEP by
tools/make-inkling-test-gguf.py (feature header, CI policy: no full-model
run outside a booked window; 16 experts / 6 used / 2 shared, one MoE
layer). INKLING_REMOTE_EXPERTS_MODEL overrides it with any Inkling GGUF, e.g.
Inkling-Small in a booked window. The MoE layer range is read from the GGUF
header (inkling.dense_block_count .. inkling.block_count - 1), never guessed.

INKLING_REMOTE_EXPERTS_CLIENT_ARGS adds arguments to BOTH expert-check runs
(e.g. "-fa off" for the masked attention path; the default is the build's
own default). INKLING_REMOTE_EXPERTS_SERVER_ARGS adds arguments to the
expert-server; it exists so the gate can be shown to bite — "--moe-form
moe_ffn" forces the llm_build_moe_ffn tail on an inkling file and must FAIL
the byte-identity Then, the same fusion-contract negative the expert-server
feature carries. A green run never sets it.

Fail closed (feature header): the Givens only resolve and record —
binaries (LLAMA_EXPERT_SERVER_BIN / LLAMA_EXPERT_CHECK_BIN or build/bin),
the generator's python deps (numpy + gguf-py), the model override. The
runner When re-checks and pytest.skip()s with the precise reason before
generating a fixture, loading a model or starting a service.

"The shared expert bank stays local" is proven, not assumed: the server
resolves tensors by the ffn_*_exps names only (it has no shexp path), and
the Then checks the bank it reports is exactly the routed expert_count of
the header — not expert_count + expert_shared_count — on a model whose
header declares shared experts. The Then also requires the remote run to
have actually gone remote (server log: client connected, >= one
EXPERT_CALL per covered layer per decoded step), so a client that silently
computed everything locally (LLAMA_EXPERTS_REMOTE_KEEP_EXPS misuse, or a
graph builder that never reaches the intercept) cannot pass as byte-exact.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from pytest_bdd import given, then, when

from steps.expert_server_steps import _binary, _run_check, _start_server

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_GENERATOR = REPO_ROOT / "tools" / "make-inkling-test-gguf.py"
GGUF_PY = REPO_ROOT / "gguf-py"

SERVER_BANK_LINE = re.compile(
    r"expert-server: (\d+) layers, n_embd (\d+), n_ff_exp (\d+), n_expert (\d+), .*form (\w+)"
)
SERVER_DONE_LINE = re.compile(r"expert-server: client done: (\d+) calls")


def _python_deps_unmet() -> str | None:
    """numpy + the in-repo gguf-py are needed to generate and read the fixture."""
    try:
        import numpy  # noqa: F401
    except ImportError:
        return "numpy is not importable (needed by tools/make-inkling-test-gguf.py)"
    if str(GGUF_PY) not in sys.path:
        sys.path.insert(0, str(GGUF_PY))
    try:
        import gguf  # noqa: F401
    except ImportError as error:
        # gguf-py pulls in pyyaml (and tqdm); name the missing module, not the package
        return f"gguf-py is not importable from {GGUF_PY}: {error}"
    return None


def _resolve_d4c(bdd_context: Any) -> None:
    """Resolve every D4c prerequisite; record only, never launch."""
    state: dict[str, Any] = {"unmet": None, "model": None}
    server = _binary("LLAMA_EXPERT_SERVER_BIN", "llama-expert-server")
    check = _binary("LLAMA_EXPERT_CHECK_BIN", "llama-expert-check")
    missing = []
    if server is None:
        missing.append("LLAMA_EXPERT_SERVER_BIN (or build/bin/llama-expert-server)")
    if check is None:
        missing.append("LLAMA_EXPERT_CHECK_BIN (or build/bin/llama-expert-check)")
    if missing:
        state["unmet"] = "unresolved binaries: " + ", ".join(missing)
    elif (deps := _python_deps_unmet()) is not None:
        state["unmet"] = deps
    override = os.environ.get("INKLING_REMOTE_EXPERTS_MODEL")
    if override:
        if Path(override).is_file():
            state["model"] = Path(override)
        elif state["unmet"] is None:
            state["unmet"] = f"INKLING_REMOTE_EXPERTS_MODEL {override} is not a readable file"
    elif not FIXTURE_GENERATOR.is_file() and state["unmet"] is None:
        state["unmet"] = f"fixture generator {FIXTURE_GENERATOR} is missing"
    state["server_bin"] = server
    state["check_bin"] = check
    bdd_context.d4c = state


def _d4c_unmet(bdd_context: Any) -> str | None:
    if not hasattr(bdd_context, "d4c"):
        _resolve_d4c(bdd_context)
    return bdd_context.d4c["unmet"]


def _inkling_header(model: Path) -> dict[str, int]:
    """The header facts the scenario keys on; a non-Inkling file is a defect."""
    from gguf import GGUFReader

    reader = GGUFReader(str(model))

    def field(name: str) -> Any:
        entry = reader.fields.get(name)
        if entry is None:
            pytest.fail(f"{model}: GGUF header has no {name}")
        return entry.contents()

    arch = field("general.architecture")
    if arch != "inkling":
        pytest.fail(f"{model}: general.architecture is {arch!r}, the D4c lane needs an inkling GGUF")
    header = {
        name: int(field(f"inkling.{name}"))
        for name in (
            "block_count",
            "dense_block_count",
            "expert_count",
            "expert_used_count",
            "expert_shared_count",
        )
    }
    if header["expert_shared_count"] <= 0:
        pytest.fail(f"{model}: header declares no shared experts; the scenario cannot prove they stay local")
    if header["dense_block_count"] >= header["block_count"]:
        pytest.fail(f"{model}: no MoE layer (dense_block_count >= block_count)")
    return header


def _generate_fixture(bdd_context: Any) -> Path:
    out = bdd_context.tmp_path / "inkling-test.gguf"
    completed = subprocess.run(
        [sys.executable, str(FIXTURE_GENERATOR), str(out)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if completed.returncode != 0 or not out.is_file():
        pytest.fail(
            f"fixture generator exited {completed.returncode}; stderr tail:\n{completed.stderr[-2000:]}"
        )
    return out


def _wait_for_server_done(log_path: Path) -> str:
    """The server prints its call count when the client disconnects; give it a moment."""
    text = ""
    for _ in range(200):
        text = log_path.read_text(encoding="utf-8")
        if SERVER_DONE_LINE.search(text):
            break
        time.sleep(0.05)
    return text


@given("the expert server hosting the routed expert bank")
def given_d4c_expert_server(bdd_context: Any) -> None:
    """Resolve binaries, python deps and the model source; recording only."""
    _resolve_d4c(bdd_context)


@given("the shared expert bank stays local")
def given_d4c_shared_local(bdd_context: Any) -> None:
    """Record the claim the Then proves from the server's reported bank."""
    bdd_context.d4c_expect_shared_local = True


@when("generation runs through the remote-expert path on CPU")
def when_d4c_generation(bdd_context: Any) -> None:
    if unmet := _d4c_unmet(bdd_context):
        pytest.skip(f"Prerequisite unmet: {unmet}")
    state = bdd_context.d4c
    bdd_context.inkling_launches = getattr(bdd_context, "inkling_launches", 0) + 1

    model = state["model"] or _generate_fixture(bdd_context)
    header = _inkling_header(model)
    moe_layers = list(range(header["dense_block_count"], header["block_count"]))
    layer_range = f"{moe_layers[0]}-{moe_layers[-1]}"

    # the shared helpers read the expert-server feature's context attributes
    bdd_context.moe_model = model
    bdd_context.expert_server_bin = state["server_bin"]
    bdd_context.expert_check_bin = state["check_bin"]
    bdd_context.expert_prompt = os.environ.get("BDD_EXPERT_PROMPT", "The capital of France is")
    bdd_context.expert_tokens = int(os.environ.get("BDD_EXPERT_TOKENS", "12"))
    client_arguments = os.environ.get("INKLING_REMOTE_EXPERTS_CLIENT_ARGS", "")
    server_arguments = os.environ.get("INKLING_REMOTE_EXPERTS_SERVER_ARGS", "")

    local = _run_check(bdd_context, remote=None, client_arguments=client_arguments, label="d4c-local")
    if local["returncode"] != 0:
        pytest.fail(f"in-process expert-check exited {local['returncode']}:\n{local['stderr'][-3000:]}")

    # --moe-form is left on auto on purpose: the server must pick the inkling
    # tail from general.architecture by itself, or an operator-facing default
    # would be wrong for this model (INKLING_REMOTE_EXPERTS_SERVER_ARGS is the
    # documented way to override it for a negative control)
    _, port, log_path = _start_server(bdd_context, layer_range, server_arguments)
    remote = _run_check(
        bdd_context,
        remote=f"127.0.0.1:{port}@{layer_range}",
        client_arguments=client_arguments,
        label="d4c-remote",
    )
    if remote["returncode"] != 0:
        pytest.fail(f"remote expert-check exited {remote['returncode']}:\n{remote['stderr'][-3000:]}")

    bdd_context.d4c_run = {
        "model": model,
        "header": header,
        "moe_layers": moe_layers,
        "local": local,
        "remote": remote,
        "server_log": _wait_for_server_done(log_path),
    }


@then("token IDs and logits are byte-exact against the in-process run")
def then_d4c_byte_exact(bdd_context: Any) -> None:
    run = getattr(bdd_context, "d4c_run", None)
    if run is None:
        pytest.fail("no remote-expert run recorded: the When must run before the gate (fail closed)")
    local, remote, header = run["local"], run["remote"], run["header"]

    # the run must have decoded something, or the comparison is vacuous
    if not local["tokens"]:
        pytest.fail(f"in-process run decoded no tokens:\n{local['stdout']}")
    local_bytes = local["logits_path"].read_bytes()
    if not local_bytes:
        pytest.fail("in-process run wrote an empty logits dump")

    # ... and the remote run must actually have gone remote, on every covered layer
    log = run["server_log"]
    if "client connected" not in log:
        pytest.fail(f"expert-server never saw the client connect:\n{log[-3000:]}")
    done = SERVER_DONE_LINE.search(log)
    if done is None:
        pytest.fail(f"expert-server never reported a completed client session:\n{log[-3000:]}")
    n_calls = int(done.group(1))
    n_steps = len(remote["tokens"])
    least = len(run["moe_layers"]) * n_steps  # one prefill + one call per decoded token, per MoE layer
    if n_calls < least:
        pytest.fail(
            f"expert-server answered {n_calls} EXPERT_CALLs, fewer than {least} "
            f"({len(run['moe_layers'])} MoE layer(s) x {n_steps} decoded steps): "
            f"some covered layer was not served remotely"
        )
    if "rejecting call" in log:
        pytest.fail(f"expert-server rejected a call:\n{log[-3000:]}")

    # the gate: byte-identity, three ways
    local_ids = [(step, token) for step, token, _ in local["tokens"]]
    remote_ids = [(step, token) for step, token, _ in remote["tokens"]]
    assert remote_ids == local_ids, f"token ids differ:\n  local  {local_ids}\n  remote {remote_ids}"
    assert remote["tokens"] == local["tokens"], (
        "per-step logits hashes differ:\n  local  "
        f"{[h for _, _, h in local['tokens']]}\n  remote {[h for _, _, h in remote['tokens']]}"
    )
    remote_bytes = remote["logits_path"].read_bytes()
    assert remote_bytes == local_bytes, (
        f"raw logits dumps differ ({len(local_bytes)} vs {len(remote_bytes)} bytes)"
    )

    # the shared expert bank stays local: the bank the server loaded is the
    # routed expert_count exactly, on a header that declares shared experts
    bank = SERVER_BANK_LINE.search(log)
    if bank is None:
        pytest.fail(f"expert-server did not report its expert bank:\n{log[-3000:]}")
    n_layers, _, _, n_expert, form = bank.groups()
    if int(n_expert) != header["expert_count"]:
        pytest.fail(
            f"expert-server serves a bank of {n_expert} experts; the routed bank is "
            f"{header['expert_count']} (+{header['expert_shared_count']} shared, which must stay local)"
        )
    if int(n_layers) != len(run["moe_layers"]):
        pytest.fail(f"expert-server loaded {n_layers} layers, expected {len(run['moe_layers'])}")
    if form != "inkling":
        pytest.fail(f"expert-server mirrored the {form!r} tail on an inkling GGUF; --moe-form auto is wrong")
