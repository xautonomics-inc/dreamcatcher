"""BDD step definitions for the Inkling feature — D2 oracle parity + D6 CI fixture.

Scope: the two @d2 parity scenarios, the @d6 @ci synthetic-fixture smoke and
the @fail-closed scenario of features/inkling.feature. The other lanes (D1,
D3, D4a/b/c) bind when their lanes implement them (see test_inkling.py).

Binding contract (feature header):
- Paths resolve from the environment — INKLING_MODEL_PATH, INKLING_ORACLE_DIR,
  plus INKLING_GREEDY_CMD / INKLING_PPL_CMD / INKLING_PPL_TEXT for the
  runners — and fail closed when unset, missing or hash-mismatched.
- Fail closed means the scenario stops before touching a model or starting a
  service. Givens only record resolution state; every runner When re-checks
  it and pytest.skip()s with the precise reason before launching anything.

Oracle artifacts (sha256 pinned in the feature header) and asserted values:
- greedy-64x8.json: per-prompt greedy token IDs with per-position logprob and
  top_logprobs (n_probs: 10).
- kld-base-4x2048.bin + ppl.log: four per-chunk perplexity values
  94.3665, 83.6386, 78.6291, 72.6183 — the final estimate IS chunk 4.
The oracle's own error bar (± 5.49154) is NOT a tolerance.

Cross-lineage envelope (D2 amendment, 2026-09-19) — no tolerances:
- The equality gate is replaced by a pre-registered envelope in the
  INKLING_ENVELOPE_JSON file. The step file pins the schema and metric
  names, never a calibration figure:
  {"metrics": {"<name>": {"bound": <float > 0>, "ceiling": <float>,
  "provenance": "<calibration run>"}}}
- Pinned metric names (ENVELOPE_METRICS): greedy_logprob_delta (max
  absolute per-position logprob deviation), token_mismatch_rate (fraction
  of positions whose token ID differs), perplexity_chunk_delta (max
  absolute per-chunk deviation), perplexity_final_delta (absolute
  deviation of the final estimate from the recorded overall perplexity).
- A bound exceeding its ceiling is surfaced as a finding by the dedicated
  Then; a structurally invalid envelope fails at the Given.
  INKLING_ENVELOPE_JSON unset/unreadable is recorded unmet and the runner
  Whens skip before launching anything — fail closed, like every runner
  variable here.

Runner contract (until the D2 binary lands these env vars are unbound and
the scenarios skip):
- INKLING_GREEDY_CMD: shell-free command template with a {model} placeholder;
  stdout must be the greedy dump in the oracle's schema (see
  _greedy_positions for the required record shape).
- INKLING_PPL_CMD: command template with {model} and {text} placeholders;
  stdout must carry llama.cpp-style "[i]value" per-chunk entries and a
  "Final estimate: PPL = X +/- Y" line.
- INKLING_ORACLE_FEATURES / INKLING_BUILD_FEATURES: comma-separated CPU
  feature sets of the oracle and runner builds. Token-for-token equality
  is asserted as a gate only when both are declared and equal; otherwise
  the comparison is recorded as a diagnostic warning and the envelope is
  the gate.
- Both run with CUDA_VISIBLE_DEVICES forced empty: the oracle binary was
  CUDA-built but run CPU-only, and its CPU feature set (AVX2, FMA, LLAMAFILE,
  REPACK) is part of the environment a parity run must reproduce. The -t 20
  thread count is the command's own responsibility.

D6 contract (@d6 @ci — the CI policy paragraph in the feature header): the
one lane that deliberately does NOT gate on INKLING_ORACLE_DIR or
INKLING_MODEL_PATH. CI has neither the D0 artifacts nor the 160 GiB model;
it smoke-tests the D1 arch plumbing and the D2 graph on the ~19 MB synthetic
fixture that tools/make-inkling-test-gguf.py generates in-step (D6 section
at the bottom of this file):
- Given: run the generator into the scenario's tmp_path and read the header
  back with the repo's gguf-py — arch, KV count, tensor count. A missing
  generator or numpy is a prerequisite (recorded unmet -> the When skips); a
  generator that exits non-zero or writes a non-inkling file is a defect
  (pytest.fail).
- When: resolve llama-perplexity (LLAMA_PERPLEXITY_BIN) and llama-cli
  (LLAMA_CLI_BIN) like LLAMA_SERVER_BIN elsewhere in this suite (env var,
  then build/bin, then PATH) and skip with the precise reason when either is
  missing — never a mock. Run both on the fixture, CPU-only
  (CUDA_VISIBLE_DEVICES forced empty), a few hundred tokens of perplexity
  and a handful of generated tokens. Nothing is asserted here; the Then is
  the gate.
- Then, fail closed on a missing run: (a) the arch LOADED — the loader's
  "loaded meta data with K key-value pairs and N tensors" line matches the
  fixture header, print-meta reports arch = inkling, llm_load_tensors
  allocated a weight buffer, no loader error line, exit 0. Load success is
  equivalent to "every tensor created": llama_model_loader::
  done_getting_tensors() throws "wrong number of tensors" whenever the arch
  creates fewer tensors than the file holds. (b) the GRAPH BUILT and a
  forward pass produced finite logits — "Final estimate: PPL ... = X" is
  present and X is finite (a NaN/Inf logit anywhere in the graph poisons
  the estimate), and llama-cli generated tokens (its timing lines report
  >= 1 prompt token and >= 1 eval run) and exited 0. (c) WITHOUT the full
  model — the fixture is under D6_MAX_FIXTURE_BYTES and is not the file
  INKLING_MODEL_PATH names.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
from pytest_bdd import given, parsers, then, when

# Metric names this step file consults in the pre-registered envelope
# (schema in the module docstring). The envelope may carry more entries;
# only these four are the D2 gate. Names are pinned here and nowhere else.
ENVELOPE_METRICS = (
    "greedy_logprob_delta",     # max absolute per-position logprob deviation
    "token_mismatch_rate",      # fraction of positions with a differing token ID
    "perplexity_chunk_delta",   # max absolute per-chunk ppl deviation
    "perplexity_final_delta",   # |final estimate - recorded overall ppl|
)

ORACLE_FILES = {
    "greedy-64x8.json": "1b5e5c4bff5b98cbe91345e5df290a804667e25ff728fbbba78b3d08cef923f0",
    "kld-base-4x2048.bin": "aeab11429a2567db2b36f210754a444ce97f6a9fa851bf0389b236129838b505",
    "ppl.log": "98f7349cb50c92500b5cd79171e0fcf34453a79abec7c7fccfd881f1f4ab4edc",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_oracle_dir(bdd_context) -> None:
    """Resolve INKLING_ORACLE_DIR and hash-verify the pinned artifacts.

    Records {"dir": Path | None, "unmet": str | None} on the context. Never
    touches a model or a service; the runner Whens do the fail-closed skip.
    """
    state: dict = {"dir": None, "unmet": None}
    raw = os.environ.get("INKLING_ORACLE_DIR")
    if not raw:
        state["unmet"] = "INKLING_ORACLE_DIR is not bound"
    elif not (directory := Path(raw)).is_dir():
        state["unmet"] = f"INKLING_ORACLE_DIR {raw} does not exist"
    else:
        for name, expected in ORACLE_FILES.items():
            path = directory / name
            if not path.is_file():
                state["unmet"] = f"INKLING_ORACLE_DIR is missing {name}"
                break
            if (actual := _sha256(path)) != expected:
                state["unmet"] = f"{name} failed hash verification: {actual}"
                break
        else:
            state["dir"] = directory
    bdd_context.inkling_oracle = state


def _oracle_unmet(bdd_context) -> str | None:
    if not hasattr(bdd_context, "inkling_oracle"):
        _resolve_oracle_dir(bdd_context)
    return bdd_context.inkling_oracle["unmet"]


def _resolve_envelope(bdd_context) -> None:
    """Resolve INKLING_ENVELOPE_JSON and validate the pre-registered envelope.

    Records {"data": dict | None, "unmet": str | None} on the context.
    Recording only — the runner Whens do the fail-closed skip before any
    launch, and the envelope Thens gate on the parsed metrics. A missing
    envelope is a prerequisite (unmet -> skip); a present-but-malformed
    envelope is a defect (pytest.fail with the schema).
    """
    state: dict = {"data": None, "unmet": None}
    raw = os.environ.get("INKLING_ENVELOPE_JSON")
    if not raw:
        default_path = Path(__file__).resolve().parent.parent / "fixtures" / "inkling-envelope.json"
        if default_path.is_file():
            raw = str(default_path)
    if not raw:
        state["unmet"] = "INKLING_ENVELOPE_JSON is not bound"
    elif not (path := Path(raw)).is_file():
        state["unmet"] = f"INKLING_ENVELOPE_JSON {raw} is not a readable file"
    else:
        try:
            data = json.loads(path.read_bytes())
        except (OSError, json.JSONDecodeError) as error:
            state["unmet"] = f"INKLING_ENVELOPE_JSON {raw} is not valid JSON: {error!r}"
        else:
            state["data"] = _validate_envelope(data, raw)
    bdd_context.inkling_envelope = state


def _validate_envelope(data, source: str) -> dict:
    """Structural validation of a pre-registered envelope (schema: docstring).

    Required shape: {"metrics": {"<name>": {"bound": float > 0,
    "ceiling": float, "provenance": str}}}. Every pinned metric in
    ENVELOPE_METRICS must be present. A bound exceeding its ceiling is
    deliberately NOT rejected here: that is a registered finding the
    dedicated Then surfaces as data, not a structural defect.
    """
    if not isinstance(data, dict) or not isinstance(data.get("metrics"), dict):
        pytest.fail(f"{source}: envelope must be {{\"metrics\": {{...}}}}")
    metrics = data["metrics"]
    for name in ENVELOPE_METRICS:
        entry = metrics.get(name)
        if not isinstance(entry, dict):
            pytest.fail(f"{source}: envelope metric {name!r} is missing or not an object")
        for field in ("bound", "ceiling"):
            value = entry.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                pytest.fail(f"{source}: envelope metric {name!r} field {field!r} must be a number")
        if not entry["bound"] > 0:
            pytest.fail(f"{source}: envelope metric {name!r} bound must be > 0")
        provenance = entry.get("provenance")
        if not isinstance(provenance, str) or not provenance:
            pytest.fail(f"{source}: envelope metric {name!r} needs a non-empty provenance")
    return data


def _envelope_unmet(bdd_context) -> str | None:
    if not hasattr(bdd_context, "inkling_envelope"):
        _resolve_envelope(bdd_context)
    return bdd_context.inkling_envelope["unmet"]


@given(
    parsers.parse(
        "a pre-registered cross-lineage envelope exists in the INKLING_ENVELOPE_JSON file"
    )
)
def given_inkling_envelope(bdd_context):
    """Resolve and structurally validate the envelope; recording only, no skip."""
    _resolve_envelope(bdd_context)


def _greedy_positions(record: dict) -> list[dict]:
    """Extract per-position entries from one greedy dump record.

    Required record shape (greedy-64x8.json): {"prompt": str, "tokens": [
    {"id": int, "logprob": float, "top_logprobs": [float, ... <= 10]}, ...]}.
    A mismatch fails loudly with the schema so the adaptation happens here
    and nowhere else when the real dump lands.
    """
    try:
        positions = record["tokens"]
        return [
            {
                "id": int(entry["id"]),
                "logprob": float(entry["logprob"]),
                "top_logprobs": [float(value) for value in entry["top_logprobs"]],
            }
            for entry in positions
        ]
    except (KeyError, TypeError, ValueError) as error:
        pytest.fail(
            "greedy dump record does not match the documented schema "
            f'({{"prompt": str, "tokens": [{{"id", "logprob", "top_logprobs"}}]}}): '
            f"{error!r}"
        )


def _load_greedy_records(payload: bytes, source: str) -> list[dict]:
    data = json.loads(payload)
    records = data.get("prompts", data) if isinstance(data, dict) else data
    if not isinstance(records, list) or not records:
        pytest.fail(f"{source}: expected a non-empty list of prompt records")
    return records


def _require(bdd_context, command_env: str) -> tuple[Path, Path, str]:
    """Fail-closed gate shared by the runner Whens, before any launch.

    Gates on every runner prerequisite in Given order — oracle directory,
    pre-registered envelope, model path, runner command — so an unset or
    unreadable envelope skips the launch exactly like an unset INKLING_*
    variable. The envelope is never optional: without it there is no gate,
    and a gate that silently degenerates into "run and compare nothing"
    would be worse than no run (D2 amendment).
    """
    if unmet := _oracle_unmet(bdd_context):
        pytest.skip(f"Prerequisite unmet: {unmet}")
    if unmet := _envelope_unmet(bdd_context):
        pytest.skip(f"Prerequisite unmet: {unmet}")
    model = getattr(bdd_context, "inkling_model_path", None)
    if model is None:
        pytest.skip("Prerequisite unmet: INKLING_MODEL_PATH is not bound to a readable file")
    command = os.environ.get(command_env)
    if not command:
        pytest.skip(f"Prerequisite unmet: {command_env} is not bound (D2 runner not wired)")
    bdd_context.inkling_launches = getattr(bdd_context, "inkling_launches", 0) + 1
    return bdd_context.inkling_oracle["dir"], model, command


def _run_command(command: str, **placeholders: str) -> str:
    argv = [part.format(**placeholders) for part in shlex.split(command)]
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    completed = subprocess.run(
        argv, capture_output=True, text=True, timeout=3600, env=environment, check=False
    )
    if completed.returncode != 0:
        pytest.fail(
            f"{argv[0]} exited {completed.returncode}; stderr tail:\n{completed.stderr[-2000:]}"
        )
    return completed.stdout


@given(parsers.parse("Inkling-Small GGUF metadata for the model named by INKLING_MODEL_PATH"))
def given_inkling_model_path(bdd_context):
    """Record the model path; the runner Whens gate on it."""
    raw = os.environ.get("INKLING_MODEL_PATH")
    path = Path(raw) if raw else None
    bdd_context.inkling_model_path = path if path and path.is_file() else None


@given(
    parsers.parse("a D0 oracle artifact directory named by INKLING_ORACLE_DIR with recorded hashes")
)
def given_inkling_oracle_dir(bdd_context):
    """Resolve and hash-verify the D0 artifacts; recording only, no skip."""
    _resolve_oracle_dir(bdd_context)


@given(
    parsers.parse(
        'the D0 oracle dump "{name}" exists in the INKLING_ORACLE_DIR directory'
    )
)
def given_inkling_dump_exists(bdd_context, name):
    """Record the dump path; the runner Whens gate on it."""
    state = {"path": None, "unmet": _oracle_unmet(bdd_context)}
    if not state["unmet"]:
        path = bdd_context.inkling_oracle["dir"] / name
        state["path"] = path if path.is_file() else None
        if state["path"] is None:
            state["unmet"] = f"oracle dump {name} is not a readable file"
    bdd_context.inkling_dump = state


@when(parsers.parse("the model runs greedy generation on CPU for the D0 fixed prompt set"))
def when_greedy_run(bdd_context):
    directory, model, command = _require(bdd_context, "INKLING_GREEDY_CMD")
    oracle_records = _load_greedy_records(
        (directory / "greedy-64x8.json").read_bytes(), "oracle greedy-64x8.json"
    )
    stdout = _run_command(command, model=str(model))
    bdd_context.inkling_greedy_oracle = oracle_records
    bdd_context.inkling_greedy_run = _load_greedy_records(stdout.encode(), "greedy runner stdout")


def _envelope_metrics(bdd_context) -> dict:
    """Parsed envelope metrics, or skip when the prerequisite is unmet.

    Gate Thens call this instead of touching the context attribute
    directly, so an envelope that failed to resolve skips the comparison
    rather than raising on a missing attribute — fail closed, same as the
    runner Whens.
    """
    if unmet := _envelope_unmet(bdd_context):
        pytest.skip(f"Prerequisite unmet: {unmet}")
    return bdd_context.inkling_envelope["data"]["metrics"]


def _envelope_fails(entry: dict, observed: float) -> str | None:
    """Band check of one observed value against one envelope entry.

    Band semantics: the metric holds when observed <= bound. Returns a
    failure report line (with the calibration provenance) or None when
    the value sits inside the band.
    """
    if observed <= entry["bound"]:
        return None
    return (
        f"observed {observed!r} > bound {entry['bound']!r} "
        f"(provenance: {entry['provenance']})"
    )


def _greedy_comparison(bdd_context) -> tuple[list[float], int, int]:
    """Align oracle vs run greedy records; return (deltas, mismatches, positions).

    Shared by the envelope Then (band metrics) and the conditional
    token-equality Then (diagnostic tally), so both steps always reason
    over the same comparison. Structural divergence — prompt-record count,
    per-prompt position count, top_logprobs length — pytest.fail()s here:
    those are shape defects, not bandable deviations. Per-rank top-logprob
    values are deliberately uncompared: no pinned envelope metric covers
    them (the D2 amendment pins max logprob delta and mismatch rate only).
    """
    oracle_records = bdd_context.inkling_greedy_oracle
    run_records = bdd_context.inkling_greedy_run
    if len(run_records) != len(oracle_records):
        pytest.fail(
            f"greedy run produced {len(run_records)} prompt records, "
            f"oracle has {len(oracle_records)}"
        )
    deltas: list[float] = []
    mismatches = 0
    positions = 0
    for index, (oracle, run) in enumerate(zip(oracle_records, run_records)):
        oracle_positions = _greedy_positions(oracle)
        run_positions = _greedy_positions(run)
        if len(run_positions) != len(oracle_positions):
            pytest.fail(
                f"prompt {index}: run generated {len(run_positions)} positions, "
                f"oracle has {len(oracle_positions)} — envelope metrics are not "
                f"computable over divergent position counts"
            )
        for position, (expected, actual) in enumerate(
            zip(oracle_positions, run_positions)
        ):
            positions += 1
            deltas.append(abs(actual["logprob"] - expected["logprob"]))
            if expected["id"] != actual["id"]:
                mismatches += 1
            if len(actual["top_logprobs"]) != len(expected["top_logprobs"]):
                pytest.fail(
                    f"prompt {index} position {position}: runner emitted "
                    f"{len(actual['top_logprobs'])} top_logprobs, oracle dump has "
                    f"{len(expected['top_logprobs'])} (structural: n_probs must "
                    f"match the oracle dump, this is not a bandable deviation)"
                )
    return deltas, mismatches, positions


def _declared_features(env_var: str) -> frozenset[str] | None:
    """Comma-separated GGML feature set from an env var; None when undeclared."""
    raw = os.environ.get(env_var)
    if not raw or not raw.strip():
        return None
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


def _record_diagnostic(message: str) -> None:
    """Surface a non-gate comparison outcome; never silent, never failing."""
    warnings.warn(message, stacklevel=2)


@then(
    parsers.parse(
        "the greedy metrics declared in the envelope sit inside their "
        "pre-registered bounds"
    )
)
def then_greedy_metrics_within_envelope(bdd_context):
    metrics = _envelope_metrics(bdd_context)
    deltas, mismatches, positions = _greedy_comparison(bdd_context)
    for name, observed in (
        ("greedy_logprob_delta", max(deltas, default=0.0)),
        ("token_mismatch_rate", mismatches / positions if positions else 0.0),
    ):
        if report := _envelope_fails(metrics[name], observed):
            pytest.fail(f"envelope metric {name}: {report}")


@then(
    parsers.parse(
        "token-for-token equality is asserted only when the build features "
        "match the oracle build (GGML_LLAMAFILE, GGML_CPU_REPACK) and is "
        "otherwise recorded as a diagnostic, not a gate"
    )
)
def then_greedy_tokens_conditional(bdd_context):
    """Feature line 80: equality gates only under proven feature parity.

    "Only when" is read strictly: the gate fires solely when BOTH feature
    sets are declared AND equal. Undeclared features degrade to diagnostic
    mode rather than failing — the envelope is the D2 gate and stands on
    its own; failing closed here would make the diagnostic branch
    unreachable and re-introduce unconditional equality by the back door.
    """
    _, mismatches, positions = _greedy_comparison(bdd_context)
    tally = f"{mismatches}/{positions} token positions differ from the oracle"
    oracle_features = _declared_features("INKLING_ORACLE_FEATURES")
    build_features = _declared_features("INKLING_BUILD_FEATURES")
    if oracle_features is None or build_features is None:
        undeclared = [
            name
            for name, value in (
                ("INKLING_ORACLE_FEATURES", oracle_features),
                ("INKLING_BUILD_FEATURES", build_features),
            )
            if value is None
        ]
        _record_diagnostic(
            f"token-for-token equality NOT asserted as a gate: "
            f"{', '.join(undeclared)} undeclared, feature parity unprovable; "
            f"diagnostic: {tally}"
        )
        return
    if build_features != oracle_features:
        _record_diagnostic(
            f"token-for-token equality NOT asserted as a gate: build features "
            f"{sorted(build_features)} != oracle features {sorted(oracle_features)}; "
            f"diagnostic: {tally}"
        )
        return
    if mismatches:
        pytest.fail(
            f"feature parity proven ({', '.join(sorted(build_features))}) but "
            f"{tally} — same-feature-set builds must match token-for-token"
        )


@when(parsers.parse("the model computes perplexity on CPU over the D0 4-chunk split"))
def when_ppl_run(bdd_context):
    directory, model, command = _require(bdd_context, "INKLING_PPL_CMD")
    text = os.environ.get("INKLING_PPL_TEXT")
    if not text or not Path(text).is_file():
        pytest.skip("Prerequisite unmet: INKLING_PPL_TEXT is not bound to a readable file")
    stdout = _run_command(command, model=str(model), text=text)
    chunks = [
        float(value)
        for _, value in sorted(
            (int(index), float(value))
            for index, value in re.findall(r"\[(\d+)\]([0-9]+\.[0-9]+)", stdout)
        )
    ]
    final = re.search(r"Final estimate: PPL = ([0-9.]+) (?:\+/-|±) ([0-9.]+)", stdout)
    if len(chunks) != 4 or final is None:
        pytest.fail(
            f"perplexity run did not yield 4 chunks + final estimate "
            f"(got {len(chunks)} chunks, final={'yes' if final else 'no'})"
        )
    bdd_context.inkling_ppl = {
        "chunks": chunks,
        "final": float(final.group(1)),
        "error_bar": float(final.group(2)),
    }


@then(
    parsers.parse(
        "the four per-chunk perplexity values fall within the envelope "
        "band around the oracle {first:g}, {second:g}, {third:g}, {fourth:g}"
    )
)
def then_ppl_chunks_within_envelope(
    bdd_context, first: float, second: float, third: float, fourth: float
):
    anchors = [first, second, third, fourth]
    entry = _envelope_metrics(bdd_context)["perplexity_chunk_delta"]
    for index, (anchor, got) in enumerate(
        zip(anchors, bdd_context.inkling_ppl["chunks"]), start=1
    ):
        delta = abs(got - anchor)
        if report := _envelope_fails(entry, delta):
            pytest.fail(
                f"chunk {index}: |{got} - {anchor}| = {delta} outside "
                f"perplexity_chunk_delta band: {report}"
            )


@then(
    parsers.parse(
        "the final chunk stays within the envelope band around the recorded "
        "overall perplexity {final:g}"
    )
)
def then_ppl_final_within_envelope(bdd_context, final: float):
    ppl = bdd_context.inkling_ppl
    entry = _envelope_metrics(bdd_context)["perplexity_final_delta"]
    for label, got in (
        ("final chunk", ppl["chunks"][-1]),
        ("run final estimate", ppl["final"]),
    ):
        delta = abs(got - final)
        if report := _envelope_fails(entry, delta):
            pytest.fail(
                f"{label} {got} vs recorded overall perplexity {final}: "
                f"|delta| = {delta} outside perplexity_final_delta band: {report}"
            )


@then(
    parsers.parse(
        "an envelope whose band exceeds its declared ceiling is surfaced as "
        "a finding, not absorbed into the envelope"
    )
)
def then_envelope_band_exceedances_surfaced(bdd_context):
    """Every entry is audited, not just the pinned four: an over-wide band
    anywhere in the envelope is a finding. Surfaced via pytest.fail carrying
    bound, ceiling and provenance — a finding that leaves the run green is
    absorption by another name."""
    metrics = _envelope_metrics(bdd_context)
    findings = [
        f"{name}: bound {entry['bound']!r} > ceiling {entry['ceiling']!r} "
        f"(provenance: {entry['provenance']})"
        for name, entry in sorted(metrics.items())
        if entry["bound"] > entry["ceiling"]
    ]
    if findings:
        pytest.fail(
            "envelope registration findings — band exceeds declared ceiling:\n"
            + "\n".join(f"  - {finding}" for finding in findings)
        )


@given(parsers.parse("that directory does not exist or fails hash verification"))
def given_fail_closed_bad_oracle_dir(bdd_context):
    """The fail-closed scenario: only provable on a host without valid artifacts."""
    if not _oracle_unmet(bdd_context):
        pytest.skip(
            "Prerequisite unmet: INKLING_ORACLE_DIR is present and hash-valid; "
            "the fail-closed path is not exercised on this host"
        )


@when(parsers.parse("the scenario attempts to bind expected values"))
def when_fail_closed_bind(bdd_context):
    _resolve_oracle_dir(bdd_context)
    if not bdd_context.inkling_oracle["unmet"]:
        pytest.fail("binding succeeded on an invalid oracle directory; fail-closed is broken")


@then(parsers.parse("the scenario aborts before loading any model or starting any service"))
def then_fail_closed_aborted(bdd_context):
    assert bdd_context.inkling_oracle["unmet"], "resolution must record the unmet prerequisite"
    launches = getattr(bdd_context, "inkling_launches", 0)
    assert launches == 0, f"{launches} model/service launch(es) happened before the abort"


# ---------------------------------------------------------------------------
# D6 — the synthetic-GGUF CI fixture (@d6 @ci). Contract in the module
# docstring. CPU-only, no GPU, no model download, no oracle: this lane
# smoke-tests that the Inkling arch loads and its graph runs, nothing more.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
D6_GENERATOR = REPO_ROOT / "tools" / "make-inkling-test-gguf.py"
D6_GGUF_PY = REPO_ROOT / "gguf-py"
D6_ARCH = "inkling"
D6_FIXTURE_NAME = "inkling-d6.gguf"
# "Without the full model": the generator writes ~19 MB; the 160 GiB model
# (or any real checkpoint) can never pass this bound.
D6_MAX_FIXTURE_BYTES = 64 * 1024 * 1024
# Context above inkling.attention.sliding_window (32) and both rel_extents
# (64 / 48) in the fixture, so the SWA mask and the bias pad column are on
# the exercised path; two chunks so the graph runs more than once.
D6_N_CTX = 256
D6_N_CHUNKS = 2
D6_N_PREDICT = 8
D6_PROMPT = "The quick brown fox jumps over the lazy dog."
D6_TIMEOUT_S = 600
# Binary name -> env override, mirroring LLAMA_SERVER_BIN (README §4).
D6_BINARIES = {
    "llama-perplexity": "LLAMA_PERPLEXITY_BIN",
    "llama-cli": "LLAMA_CLI_BIN",
}
# Loader / graph failure lines. Any hit fails the load assertion even when
# the process somehow exited 0.
D6_LOAD_ERROR_PATTERNS = (
    r"error loading model",
    r"failed to load model",
    r"wrong number of tensors",
    r"tensor '[^']+' (?:not found|has wrong shape)",
    r"unknown (?:model )?architecture",
    r"error: input is empty",
    r"GGML_ASSERT",
)


def _find_binary(name: str, env_var: str) -> tuple[str | None, str | None]:
    """Locate a built binary: env override, then build/bin, build, PATH.

    Returns (path, unmet reason). An env override that names a missing file
    is reported as such rather than silently falling through — an explicitly
    bound but absent binary is a misconfiguration the skip reason must name.
    """
    env_bin = os.environ.get(env_var)
    if env_bin:
        if Path(env_bin).is_file() and os.access(env_bin, os.X_OK):
            return env_bin, None
        return None, f"{env_var}={env_bin} is not an executable file"
    for candidate in (REPO_ROOT / "build" / "bin" / name, REPO_ROOT / "build" / name):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate), None
    if which_bin := shutil.which(name):
        return which_bin, None
    return None, f"{name} binary not found. Build with cmake or set {env_var}."


def _gguf_string(field) -> str:
    return bytes(field.parts[field.data[0]]).decode("utf-8")


def _gguf_scalar(field) -> int:
    return int(field.parts[field.data[0]][0])


def _d6_read_header(path: Path) -> dict:
    """Read arch / KV count / tensor names back with the repo's gguf-py.

    KV count is the file's own GGUF.kv_count, not len(reader.fields): the
    reader adds three GGUF.* pseudo-fields (version, tensor_count,
    kv_count) that the C++ loader does not report.
    """
    if str(D6_GGUF_PY) not in sys.path:
        sys.path.insert(0, str(D6_GGUF_PY))
    try:
        from gguf import GGUFReader
    except ImportError as error:  # the generator imports the same package from the same path
        pytest.fail(f"gguf-py is not importable from {D6_GGUF_PY} after the generator ran: {error!r}")
    reader = GGUFReader(str(path))
    return {
        "arch": _gguf_string(reader.fields["general.architecture"]),
        "n_kv": _gguf_scalar(reader.fields["GGUF.kv_count"]),
        "n_tensors": _gguf_scalar(reader.fields["GGUF.tensor_count"]),
        "tensor_names": [tensor.name for tensor in reader.tensors],
    }


@given(parsers.parse("a tiny synthetic Inkling GGUF fixture generated by the D6 generator"))
def given_d6_fixture(bdd_context):
    """Generate the fixture in-step and record its header; no model, no service.

    Records {"path", "size", "arch", "n_kv", "n_tensors", "tensor_names",
    "unmet"} on bdd_context.inkling_d6_fixture. Prerequisites (generator,
    numpy) are recorded unmet for the When to skip on; a generator that
    fails or writes the wrong arch is a defect and fails here.
    """
    state: dict = {"path": None, "unmet": None}
    if not D6_GENERATOR.is_file():
        state["unmet"] = f"D6 generator {D6_GENERATOR} is missing"
    else:
        try:
            import numpy  # noqa: F401  (the generator's only third-party dependency)
        except ImportError:
            state["unmet"] = "numpy is not importable (the D6 generator needs it)"
    if state["unmet"] is None:
        out = bdd_context.tmp_path / D6_FIXTURE_NAME
        completed = subprocess.run(
            [sys.executable, str(D6_GENERATOR), str(out)],
            capture_output=True, text=True, timeout=D6_TIMEOUT_S,
            cwd=bdd_context.tmp_path, check=False,
        )
        if completed.returncode != 0 or not out.is_file():
            pytest.fail(
                f"D6 generator exited {completed.returncode} without writing {out}; "
                f"stderr tail:\n{completed.stderr[-2000:]}"
            )
        header = _d6_read_header(out)
        if header["arch"] != D6_ARCH:
            pytest.fail(
                f"D6 generator wrote general.architecture = {header['arch']!r}, expected {D6_ARCH!r}"
            )
        if header["n_tensors"] != len(header["tensor_names"]) or header["n_tensors"] == 0:
            pytest.fail(
                f"D6 fixture header is inconsistent: GGUF.tensor_count {header['n_tensors']} vs "
                f"{len(header['tensor_names'])} tensor infos"
            )
        state.update(header, path=out, size=out.stat().st_size)
    bdd_context.inkling_d6_fixture = state


def _d6_fixture_or_skip(bdd_context) -> dict:
    state = getattr(bdd_context, "inkling_d6_fixture", None)
    if state is None:
        pytest.skip("Prerequisite unmet: the D6 fixture Given did not run")
    if state["unmet"]:
        pytest.skip(f"Prerequisite unmet: {state['unmet']}")
    return state


def _d6_ppl_text() -> str:
    """Deterministic ASCII text with comfortably more than 2 × D6_N_CTX tokens.

    The fixture vocab is byte-level (one token per ASCII byte; its only
    merges join non-ASCII byte tokens), so byte count is a lower bound on
    the token count. llama-perplexity refuses fewer than 2 × n_ctx tokens.
    """
    sentence = (
        "Inkling is a hybrid attention and short-convolution mixture-of-experts "
        "architecture; this synthetic fixture only checks that its graph runs. "
    )
    lines = []
    while sum(len(line) for line in lines) < 8 * 2 * D6_N_CTX:
        lines.append(f"{len(lines):04d} {sentence}\n")
    return "".join(lines)


def _d6_run(bdd_context, name: str, argv: list[str]) -> dict:
    """Run one smoke binary CPU-only from tmp_path and capture everything.

    Counts as a model launch (inkling_launches). Never asserts — the Then
    is the gate — but persists stdout/stderr under BDD_INKLING_D6_LOG_DIR
    when it is set so a CI pipeline keeps the loader and graph logs as
    artifacts.
    """
    bdd_context.inkling_launches = getattr(bdd_context, "inkling_launches", 0) + 1
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    try:
        completed = subprocess.run(
            argv, capture_output=True, text=True, errors="replace",
            timeout=D6_TIMEOUT_S, env=environment, cwd=bdd_context.tmp_path,
            stdin=subprocess.DEVNULL, check=False,
        )
        result = {"argv": argv, "returncode": completed.returncode,
                  "stdout": completed.stdout, "stderr": completed.stderr}
    except subprocess.TimeoutExpired as error:
        result = {"argv": argv, "returncode": None,
                  "stdout": (error.stdout or b"").decode("utf-8", "replace") if isinstance(error.stdout, bytes) else (error.stdout or ""),
                  "stderr": f"timed out after {D6_TIMEOUT_S} s\n"}
    log_dir = os.environ.get("BDD_INKLING_D6_LOG_DIR")
    if log_dir:
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{name}.stdout.log").write_text(result["stdout"])
        (directory / f"{name}.stderr.log").write_text(
            "$ " + shlex.join(argv) + f"\n(exit {result['returncode']})\n\n" + result["stderr"]
        )
    return result


@when(parsers.parse("the CI job loads the architecture and runs a graph smoke test on CPU"))
def when_d6_smoke(bdd_context):
    fixture = _d6_fixture_or_skip(bdd_context)
    binaries: dict[str, str] = {}
    for name, env_var in D6_BINARIES.items():
        path, unmet = _find_binary(name, env_var)
        if path is None:
            pytest.skip(f"Prerequisite unmet: {unmet}")
        binaries[name] = path
    text = bdd_context.tmp_path / "inkling-d6-ppl.txt"
    text.write_text(_d6_ppl_text())
    model = str(fixture["path"])
    perplexity = _d6_run(bdd_context, "llama-perplexity", [
        binaries["llama-perplexity"], "-m", model, "-f", str(text),
        "-c", str(D6_N_CTX), "-b", str(D6_N_CTX), "--chunks", str(D6_N_CHUNKS),
    ])
    cli = _d6_run(bdd_context, "llama-cli", [
        binaries["llama-cli"], "-m", model, "-p", D6_PROMPT, "-n", str(D6_N_PREDICT),
        "--temp", "0", "--ignore-eos", "--no-warmup", "--simple-io", "--no-display-prompt",
    ])
    bdd_context.inkling_d6_run = {"binaries": binaries, "perplexity": perplexity, "cli": cli}


def _d6_first_match(patterns, text: str) -> str | None:
    for pattern in patterns:
        if match := re.search(pattern, text):
            line_start = text.rfind("\n", 0, match.start()) + 1
            line_end = text.find("\n", match.end())
            return text[line_start: line_end if line_end != -1 else None].strip()
    return None


def _d6_exit_line(name: str, result: dict) -> str | None:
    if result["returncode"] == 0:
        return None
    return (
        f"{name} exited {result['returncode']}; stderr tail:\n"
        f"{result['stderr'][-1500:]}"
    )


@then(parsers.parse("the load succeeds and generation completes without the full model"))
def then_d6_load_and_generation(bdd_context):
    """The D6 gate: (a) arch loaded, (b) graph ran to finite logits and
    generated tokens, (c) on the synthetic fixture, not the full model."""
    run = getattr(bdd_context, "inkling_d6_run", None)
    if not run:
        pytest.fail("no D6 smoke run recorded: the When must run the binaries before this gate (fail closed)")
    fixture = bdd_context.inkling_d6_fixture
    failures: list[str] = []

    # (a) the Inkling arch loads: metadata read, every tensor created.
    ppl = run["perplexity"]
    load_log = ppl["stderr"] + "\n" + ppl["stdout"]
    loaded = re.search(r"loaded meta data with (\d+) key-value pairs and (\d+) tensors", load_log)
    if loaded is None:
        failures.append("loader never reported 'loaded meta data with ... tensors' (the file was not opened as a GGUF)")
    else:
        n_kv, n_tensors = int(loaded.group(1)), int(loaded.group(2))
        if (n_kv, n_tensors) != (fixture["n_kv"], fixture["n_tensors"]):
            failures.append(
                f"loader read {n_kv} key-value pairs / {n_tensors} tensors, fixture header has "
                f"{fixture['n_kv']} / {fixture['n_tensors']}"
            )
    if re.search(rf"arch\s*=\s*{D6_ARCH}\b", load_log) is None:
        failures.append(f"print-meta never reported 'arch = {D6_ARCH}' (LLM_ARCH_INKLING not resolved)")
    if re.search(r"llm_load_tensors:.*buffer size\s*=\s*[0-9.]+ MiB", load_log) is None:
        failures.append("llm_load_tensors never reported a weight buffer (tensors were not created/allocated)")
    if error_line := _d6_first_match(D6_LOAD_ERROR_PATTERNS, load_log):
        failures.append(f"loader/graph error: {error_line}")
    if exit_line := _d6_exit_line("llama-perplexity", ppl):
        failures.append(exit_line)

    # (b) the graph builds and a forward pass produces finite logits.
    estimate = re.search(
        r"Final estimate: PPL over (\d+) chunks for n_ctx=(\d+) = ([-+]?(?:[0-9.]+(?:[eE][-+]?\d+)?|nan|inf))",
        ppl["stdout"],
    )
    if estimate is None:
        failures.append("no 'Final estimate: PPL ...' line: the forward pass never produced a perplexity")
    else:
        value = float(estimate.group(3))
        if not math.isfinite(value) or value <= 0:
            failures.append(f"perplexity {estimate.group(3)} is not a finite positive number: the graph produced non-finite logits")
        if int(estimate.group(1)) != D6_N_CHUNKS or int(estimate.group(2)) != D6_N_CTX:
            failures.append(
                f"perplexity ran {estimate.group(1)} chunks at n_ctx={estimate.group(2)}, "
                f"expected {D6_N_CHUNKS} at {D6_N_CTX}"
            )
        else:
            _record_diagnostic(
                f"D6 fixture perplexity = {value:.4f} over {D6_N_CHUNKS}×{D6_N_CTX} tokens "
                f"(diagnostic: finite is the gate, the value is random-weight noise)"
            )

    # (b, generation) llama-cli decodes a handful of tokens and exits cleanly.
    cli = run["cli"]
    cli_log = cli["stderr"] + "\n" + cli["stdout"]
    prompt_eval = re.search(r"prompt eval time\s*=\s*[0-9.]+ ms /\s*(\d+) tokens", cli_log)
    if prompt_eval is None or int(prompt_eval.group(1)) < 1:
        failures.append("llama-cli reported no prompt tokens evaluated (no forward pass ran)")
    eval_runs = re.search(r"(?<!prompt )eval time\s*=\s*[0-9.]+ ms /\s*(\d+) (?:runs|tokens)", cli_log)
    if eval_runs is None or int(eval_runs.group(1)) < 1:
        failures.append(f"llama-cli reported no eval runs: generation did not complete ({D6_N_PREDICT} tokens requested)")
    if error_line := _d6_first_match(D6_LOAD_ERROR_PATTERNS, cli_log):
        failures.append(f"llama-cli error: {error_line}")
    if exit_line := _d6_exit_line("llama-cli", cli):
        failures.append(exit_line)

    # (c) without the full model.
    if fixture["size"] > D6_MAX_FIXTURE_BYTES:
        failures.append(
            f"fixture is {fixture['size']} bytes, over the {D6_MAX_FIXTURE_BYTES} byte D6 bound — "
            f"this is not the tiny synthetic fixture"
        )
    full_model = getattr(bdd_context, "inkling_model_path", None)
    if full_model is not None and Path(full_model).resolve() == Path(fixture["path"]).resolve():
        failures.append("the smoke ran on the file INKLING_MODEL_PATH names, not the in-step synthetic fixture")

    if failures:
        pytest.fail("D6 synthetic-fixture smoke failed:\n" + "\n".join(f"  - {line}" for line in failures))
