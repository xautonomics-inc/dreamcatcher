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

Oracle artifacts (sha256 pinned in the feature header):
- greedy-64x8.json: per-prompt greedy token IDs with per-position logprob and
  top_logprobs (n_probs: 10); no full logit vectors, so KLD is not computable
  from it — the greedy scenario observes top-1 agreement only.
- kld-base-4x2048.bin: the banded oracle's base logits over the D0 4-chunk
  split — the --kl-divergence base the KLD / top-1 scenario runs against.
- ppl.log: per-chunk perplexity 94.3665, 83.6386, 78.6291, 72.6183 (the
  final estimate IS chunk 4); recorded, not gated. The oracle's own error
  bar (± 5.49154) is NOT a tolerance.

Cross-lineage gate rule (candreev-blessed, 2026-09-19; the verbatim 3-part
rule is in the feature header) — no tolerances, an envelope:
- The equality gate is replaced by a pre-registered envelope in the
  INKLING_ENVELOPE_JSON file. The step file pins the schema, the metric
  names and the ×2 multiplier (WORKING_BAND_MULTIPLIER); the calibration
  figures live in the feature header's BAND record and the envelope
  registration must match it:
  {"metrics": {"<name>": {"drift": <float > 0>, "ceiling": <float>,
  "ceiling_ci": <float >= 0, optional, default 0>,
  "band": "drift_x2" | "masked_ceiling",
  "finding": "<text, required for masked_ceiling>",
  "provenance": "<calibration run>"}}}
  drift      = the measured banded-vs-banded drift of that metric
  ceiling    = the masked-path ceiling: the reference's own masked path vs
               the banded oracle, on the same metric; ceiling_ci its ± as
               printed by the calibration run
- band "drift_x2": working band = drift × WORKING_BAND_MULTIPLIER (rule 1).
  Rule 2 (band not under the ceiling) and rule 3 (band wider than half the
  masked band) are surfaced as findings via pytest.fail — by the gate Then
  for the metrics it checks (the band is capped, never clamped) and by the
  dedicated audit Then for every registered entry.
- band "masked_ceiling" (the practical D2/D3 gate, BAND record 2026-09-19,
  pending noah/emma's final encode): working band = ceiling + ceiling_ci —
  "at least as good as the reference's own masked path, within CI". Legal
  ONLY when drift × 2 actually fires rule 2 or 3 (otherwise the entry must
  be drift_x2) and only with a non-empty recorded finding, which the gate
  and audit Thens re-derive from the numbers and re-emit as a warning on
  every run. A masked_ceiling entry with no finding, or one whose drift × 2
  sits under half the ceiling, is pytest.fail()ed — a finding that is
  neither recorded nor visible is absorption by another name.
- Gate: the observed metric is <= the working band.
- Pinned metric names (ENVELOPE_METRICS), both percentages and distances
  from the banded oracle (0 = identical), straight from the KL-divergence
  summary: top1_disagreement (100 − "Same top p", the percentage of
  positions whose argmax differs from the banded oracle), rms_dp ("RMS Δp",
  the RMS token-probability deviation in %).
- A structurally invalid envelope fails at the Given. INKLING_ENVELOPE_JSON
  unset/unreadable is recorded unmet and the runner Whens skip before
  launching anything — fail closed, like every runner variable here.
- Every scenario that binds the gate Then records its observation on
  bdd_context.inkling_observed ({metric name: value}) in its When; the gate
  Then fails closed when nothing was recorded. The @d3 lane binds the same
  Then when its steps land.

Runner contract (until the D2 binary lands these env vars are unbound and
the scenarios skip):
- INKLING_GREEDY_CMD: shell-free command template with {model} and
  {oracle_dir} placeholders; stdout must be the greedy dump in the oracle's
  schema (see _greedy_positions for the record shape). tools/
  inkling-greedy-dump.py produces it from any llama-server binary, driving
  the D0 prompt set (read from {oracle_dir}/greedy-64x8.json, prompts only)
  with the D0 request body, e.g.
    INKLING_GREEDY_CMD='python3 tools/inkling-greedy-dump.py --server BIN
      --model {model} --prompts {oracle_dir}/greedy-64x8.json
      --server-args "-ngl 0 -t 20 -c 4096 -np 1 -fa off"'
  The oracle was recorded through the same endpoint and body (masked path
  is -fa off; the D0 recording itself ran the reference's banded default).
- INKLING_PPL_CMD: command template with {model}, {text} and {base}
  placeholders ({base} = the sha-pinned kld-base-4x2048.bin, i.e. a
  llama-perplexity --kl-divergence run); stdout must carry the llama.cpp
  KL-divergence summary lines "RMS Δp    : X ± Y %" and "Same top p: Z ± W %".
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

# Gate rule 1 (feature header): working band = measured banded-vs-banded
# drift × 2. The multiplier is pre-registered here, fixed before any
# calibration number lands, and never read from the envelope.
WORKING_BAND_MULTIPLIER = 2

# Metric names this step file consults in the pre-registered envelope
# (schema in the module docstring). Both are percentages and distances from
# the banded D0 oracle (0 = identical), so drift, band and ceiling read the
# same way for each. The envelope may carry more entries (every entry is
# audited); only these two are the gate. Names are pinned here and nowhere
# else.
ENVELOPE_METRICS = (
    "top1_disagreement",  # 100 - top-1 agreement %: positions whose argmax differs
    "rms_dp",             # RMS token-probability deviation % vs the banded base logits
)

# Band modes an envelope entry may register (schema in the module docstring).
BAND_DRIFT_X2 = "drift_x2"            # rule 1: drift × WORKING_BAND_MULTIPLIER
BAND_MASKED_CEILING = "masked_ceiling"  # practical gate: ceiling + ceiling_ci, finding recorded
BAND_MODES = (BAND_DRIFT_X2, BAND_MASKED_CEILING)

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

    Required shape: {"metrics": {"<name>": {"drift": float > 0,
    "ceiling": float, "ceiling_ci"?: float >= 0, "band": BAND_MODES,
    "finding"?: str, "provenance": str}}}. Every pinned metric in
    ENVELOPE_METRICS must be present, and every entry — pinned or not —
    must carry the full schema, because the audit Then reads them all.
    Rules 2 and 3 (working band vs ceiling) are deliberately NOT applied
    here: those are registered findings the gate and audit Thens surface as
    data, not structural defects.
    """
    if not isinstance(data, dict) or not isinstance(data.get("metrics"), dict):
        pytest.fail(f"{source}: envelope must be {{\"metrics\": {{...}}}}")
    metrics = data["metrics"]
    for name in ENVELOPE_METRICS:
        if name not in metrics:
            pytest.fail(f"{source}: envelope is missing the pinned metric {name!r}")
    for name, entry in metrics.items():
        if not isinstance(entry, dict):
            pytest.fail(f"{source}: envelope metric {name!r} is not an object")
        for field in ("drift", "ceiling"):
            value = entry.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                pytest.fail(f"{source}: envelope metric {name!r} field {field!r} must be a number")
        if not entry["drift"] > 0:
            pytest.fail(
                f"{source}: envelope metric {name!r} drift must be > 0 (a zero working "
                f"band cannot gate a cross-lineage run; a drift measured at exactly 0 is "
                f"a finding to raise, not a value to register)"
            )
        ceiling_ci = entry.setdefault("ceiling_ci", 0)
        if isinstance(ceiling_ci, bool) or not isinstance(ceiling_ci, (int, float)) or ceiling_ci < 0:
            pytest.fail(f"{source}: envelope metric {name!r} ceiling_ci must be a number >= 0")
        if entry.get("band") not in BAND_MODES:
            pytest.fail(f"{source}: envelope metric {name!r} band must be one of {BAND_MODES}")
        finding = entry.get("finding", "")
        if not isinstance(finding, str):
            pytest.fail(f"{source}: envelope metric {name!r} finding must be a string")
        if entry["band"] == BAND_MASKED_CEILING and not finding.strip():
            pytest.fail(
                f"{source}: envelope metric {name!r} registers band {BAND_MASKED_CEILING!r} "
                f"without a recorded finding — the rule-2/3 finding must be recorded, not absorbed"
            )
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

    Record shape = the D0 recording's (greedy-64x8.json), i.e. what llama-server
    /completion returns for the D0 request body and what tools/
    inkling-greedy-dump.py writes: {"prompt": str, "tokens": [int, ...],
    "completion_probabilities": [{"id": int, "logprob": float,
    "top_logprobs": [{"id": int, "logprob": float, ...}, ... <= n_probs]},
    ...]}. The per-position id, logprob and the top_logprobs logprob values are
    what the steps compare; "tokens" (when present) must agree with the
    per-position ids, or the record is inconsistent. A mismatch fails loudly
    with the schema so the adaptation happens here and nowhere else.
    """
    try:
        positions = [
            {
                "id": int(entry["id"]),
                "logprob": float(entry["logprob"]),
                "top_logprobs": [float(top["logprob"]) for top in entry["top_logprobs"]],
            }
            for entry in record["completion_probabilities"]
        ]
        tokens = record.get("tokens")
    except (KeyError, TypeError, ValueError) as error:
        pytest.fail(
            "greedy dump record does not match the D0 recording schema "
            '({"prompt": str, "tokens": [int], "completion_probabilities": '
            '[{"id", "logprob", "top_logprobs": [{"id", "logprob"}]}]}): '
            f"{error!r}"
        )
    if not positions:
        pytest.fail("greedy dump record holds no positions (empty completion_probabilities)")
    if tokens is not None:
        try:
            ids = [int(token) for token in tokens]
        except (TypeError, ValueError) as error:
            pytest.fail(f"greedy dump record: tokens[] is not a list of ids: {error!r}")
        if ids != [entry["id"] for entry in positions]:
            pytest.fail(
                "greedy dump record is inconsistent: tokens[] disagrees with "
                "completion_probabilities[].id"
            )
    return positions


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


RUNNER_TIMEOUT_S = float(os.environ.get("INKLING_RUNNER_TIMEOUT", "7200"))


def _run_command_full(command: str, **placeholders: str) -> tuple[str, str]:
    """Run a shell-free command template CPU-only; (stdout, stderr), fail on non-zero.

    Placeholders are substituted per shlex token ({model}, {text}, {base},
    {oracle_dir}); an unknown placeholder in the template fails loudly rather
    than launching a mangled command.
    """
    try:
        argv = [part.format(**placeholders) for part in shlex.split(command)]
    except (KeyError, IndexError, ValueError) as error:
        pytest.fail(f"runner template {command!r} uses a placeholder this step does not provide: {error!r}")
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    completed = subprocess.run(
        argv, capture_output=True, text=True, errors="replace", timeout=RUNNER_TIMEOUT_S,
        env=environment, check=False,
    )
    if completed.returncode != 0:
        pytest.fail(
            f"{argv[0]} exited {completed.returncode}; stderr tail:\n{completed.stderr[-2000:]}"
        )
    return completed.stdout, completed.stderr


def _run_command(command: str, **placeholders: str) -> str:
    return _run_command_full(command, **placeholders)[0]


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
    stdout = _run_command(command, model=str(model), oracle_dir=str(directory))
    bdd_context.inkling_greedy_oracle = oracle_records
    bdd_context.inkling_greedy_run = _load_greedy_records(stdout.encode(), "greedy runner stdout")
    # Autoregressive greedy compounds after the first divergence, so its positional
    # top-1 is NOT a cross-lineage envelope metric — the @d2 KLD scenario (teacher-forced,
    # non-compounding) owns the parity gate. This scenario asserts greedy runs COHERENTLY
    # (then_greedy_coherent); the divergence vs the oracle is recorded as a diagnostic.
    deltas, mismatches, positions = _greedy_comparison(bdd_context)
    if positions:
        _record_diagnostic(
            f"greedy top-1 divergence vs the banded oracle = {mismatches}/{positions} "
            f"positions ({100.0 * mismatches / positions:.2f}%), max |dlogprob| = "
            f"{max(deltas):.6g} (diagnostic: greedy compounds cross-lineage; not a gated metric)"
        )


@then(parsers.parse("greedy generation completes on every prompt with finite, well-formed logprobs"))
def then_greedy_coherent(bdd_context):
    """@d2 greedy gate (redefined): greedy must run coherently on the fork.

    Autoregressive greedy compounds cross-lineage, so token-parity vs the reference is
    not the gate here (the @d2 KLD scenario owns parity). This asserts the fork actually
    generates: every prompt yields well-formed positions with finite logprobs. The top-1
    divergence vs the oracle is recorded as a diagnostic in the When.
    """
    run_records = getattr(bdd_context, "inkling_greedy_run", None)
    if not run_records:
        pytest.fail("no greedy run recorded: the When must run greedy generation before this gate (fail closed)")
    failures: list[str] = []
    for index, record in enumerate(run_records):
        positions = _greedy_positions(record)
        for pos, entry in enumerate(positions):
            if not math.isfinite(entry["logprob"]):
                failures.append(f"prompt {index} pos {pos}: non-finite logprob {entry['logprob']}")
            if not entry["top_logprobs"] or not all(math.isfinite(v) for v in entry["top_logprobs"]):
                failures.append(f"prompt {index} pos {pos}: empty or non-finite top_logprobs")
    if failures:
        pytest.fail("greedy generation was not coherent:\n  " + "\n  ".join(failures[:10]))


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


def _rule1_band(entry: dict) -> float:
    """Gate rule 1: drift × the pinned multiplier — always computed from the
    registered drift, whatever band mode the entry gates on."""
    return entry["drift"] * WORKING_BAND_MULTIPLIER


def _working_band(entry: dict) -> float:
    """The band the gate compares against, per the registered band mode."""
    if entry["band"] == BAND_MASKED_CEILING:
        return entry["ceiling"] + entry["ceiling_ci"]
    return _rule1_band(entry)


def _band_findings(name: str, entry: dict) -> list[str]:
    """Gate rules 2 and 3 for one envelope entry, as finding lines.

    Always evaluated on the rule-1 band (drift × 2), never on the practical
    band — the finding is about the pre-registered rule.
    Rule 2: the working band must sit under the masked-path ceiling.
    Rule 3: a working band wider than half the masked band is a finding.
    A band failing rule 2 also fails rule 3; only the rule-2 line is
    emitted then. Findings come back as data; _surface_band_findings
    decides between pytest.fail and a re-emitted recorded finding.
    """
    band, ceiling = _rule1_band(entry), entry["ceiling"]
    detail = (
        f"drift {entry['drift']!r} × {WORKING_BAND_MULTIPLIER} = working band {band!r}, "
        f"masked-path ceiling {ceiling!r}; provenance: {entry['provenance']}"
    )
    if not band < ceiling:
        return [f"{name}: rule 2 — working band does not sit under the masked-path ceiling ({detail})"]
    if band > ceiling / 2:
        return [f"{name}: rule 3 — working band is wider than half the masked band ({detail})"]
    return []


def _surface_band_findings(name: str, entry: dict) -> None:
    """Surface rule-2/3 findings for one entry, per its band mode.

    drift_x2: any finding pytest.fail()s — the band is capped by the
    ceiling, never clamped to it.
    masked_ceiling: the finding MUST exist (else the practical band is not
    justified and the entry must go back to drift_x2) and is re-emitted as
    a warning on every run alongside the recorded finding text, so it stays
    visible without blocking the practical gate (BAND record 2026-09-19,
    pending noah/emma's final encode).
    """
    findings = _band_findings(name, entry)
    if entry["band"] == BAND_DRIFT_X2:
        if findings:
            pytest.fail(
                "working band is not capped by the masked-path ceiling — a finding, not a gate:\n"
                + "\n".join(f"  - {finding}" for finding in findings)
            )
        return
    if not findings:
        pytest.fail(
            f"{name}: registered band {BAND_MASKED_CEILING!r} is not justified — drift × "
            f"{WORKING_BAND_MULTIPLIER} = {_rule1_band(entry)!r} sits under half the masked-path "
            f"ceiling {entry['ceiling']!r}; register band {BAND_DRIFT_X2!r} (rule 1 applies)"
        )
    _record_diagnostic(
        f"{name}: gating on the masked-path ceiling ({_working_band(entry)!r}); "
        f"recorded finding, not absorbed: {entry['finding']} | "
        + "; ".join(findings)
    )


def _envelope_fails(entry: dict, observed: float) -> str | None:
    """Band check of one observed distance against one envelope entry.

    Band semantics: the metric holds when observed <= working band (per
    the registered band mode). Returns a failure report line (with the
    calibration provenance) or None when the value sits inside the band.
    """
    band = _working_band(entry)
    if observed <= band:
        return None
    return (
        f"observed {observed!r} > working band {band!r} "
        f"(band mode {entry['band']}, drift {entry['drift']!r}, "
        f"ceiling {entry['ceiling']!r} ± {entry['ceiling_ci']!r}; "
        f"provenance: {entry['provenance']})"
    )


def _greedy_comparison(bdd_context) -> tuple[list[float], int, int]:
    """Align oracle vs run greedy records; return (deltas, mismatches, positions).

    Shared by the greedy coherence When (diagnostic divergence) and the
    conditional token-equality Then (diagnostic tally). A prompt-record count
    mismatch is a structural harness error and fails here; per-prompt position
    counts may differ (autoregressive greedy can EOS early / diverge
    cross-lineage) and are compared over the common prefix. Per-rank
    top-logprob values are uncompared.
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
        # Autoregressive greedy can EOS early or diverge cross-lineage, making the run
        # shorter than the oracle; compare the common prefix (an early stop or divergence
        # is captured as a mismatch) instead of failing on the length gap. Structural
        # well-formedness is gated by then_greedy_coherent, not here.
        for position, (expected, actual) in enumerate(
            zip(oracle_positions, run_positions)
        ):
            positions += 1
            deltas.append(abs(actual["logprob"] - expected["logprob"]))
            if expected["id"] != actual["id"]:
                mismatches += 1
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
        "the KLD / top-1 agreement against the banded D0 oracle stays inside the "
        "pre-registered working band, capped by the masked-path ceiling"
    )
)
def then_agreement_within_working_band(bdd_context):
    """The cross-lineage gate, shared by the @d2 scenarios and the @d3 lane.

    Gates every metric the scenario's When recorded on inkling_observed
    (top1_disagreement for greedy; top1_disagreement + rms_dp for the
    kld-base run). "Capped by the masked-path ceiling" means a drift_x2
    band that fails rule 2 or 3 never gates: it is surfaced as a finding
    first, not clamped to the ceiling; a masked_ceiling entry gates on the
    ceiling only with its finding recorded and re-emitted. No recorded
    observation fails closed — a gate that compares nothing is not a gate.
    """
    metrics = _envelope_metrics(bdd_context)
    observed = getattr(bdd_context, "inkling_observed", None)
    if not observed:
        pytest.fail(
            "no KLD / top-1 observation recorded: the scenario's When must set "
            "bdd_context.inkling_observed before the envelope gate (fail closed)"
        )
    for name, value in sorted(observed.items()):
        if name not in ENVELOPE_METRICS:
            pytest.fail(f"observed metric {name!r} is not a pinned envelope metric {ENVELOPE_METRICS}")
        entry = metrics[name]
        _surface_band_findings(name, entry)
        if report := _envelope_fails(entry, value):
            pytest.fail(f"envelope metric {name}: {report}")


@then(
    parsers.parse(
        "token-for-token equality is asserted only when the build features "
        "match the oracle build (GGML_LLAMAFILE, GGML_CPU_REPACK) and is "
        "otherwise recorded as a diagnostic, not a gate"
    )
)
def then_greedy_tokens_conditional(bdd_context):
    """Feature: token-for-token equality gates only under proven feature parity.

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


@when(
    parsers.parse(
        "the model computes KL-divergence on CPU over the D0 4-chunk split "
        "against the banded oracle base"
    )
)
def when_kld_run(bdd_context):
    directory, model, command = _require(bdd_context, "INKLING_PPL_CMD")
    text = os.environ.get("INKLING_PPL_TEXT")
    if not text or not Path(text).is_file():
        pytest.skip("Prerequisite unmet: INKLING_PPL_TEXT is not bound to a readable file")
    base = directory / "kld-base-4x2048.bin"
    stdout = _run_command(command, model=str(model), text=text, base=str(base))
    # llama-perplexity --kl-divergence summary lines. Anything else fails
    # loudly with the contract so the adaptation happens here and nowhere
    # else when the D2 runner lands.
    rms = re.search(r"RMS\s+(?:Δp|dp)\s*:\s*([0-9]+(?:\.[0-9]+)?)", stdout)
    top1 = re.search(r"Same top p:\s*([0-9]+(?:\.[0-9]+)?)", stdout)
    if rms is None or top1 is None:
        pytest.fail(
            'KL-divergence run did not yield the summary lines "RMS Δp    : X ± Y %" and '
            f'"Same top p: Z ± W %" (RMS line: {"yes" if rms else "no"}, top-1 line: '
            f'{"yes" if top1 else "no"}); INKLING_PPL_CMD must run --kl-divergence '
            "against the {base} placeholder"
        )
    bdd_context.inkling_observed = {
        "top1_disagreement": 100.0 - float(top1.group(1)),
        "rms_dp": float(rms.group(1)),
    }


@then(
    parsers.parse(
        "a working band that is not under the masked-path ceiling, or wider "
        "than half the masked band, is surfaced as a finding, not absorbed"
    )
)
def then_working_band_findings_surfaced(bdd_context):
    """Gate rules 2 and 3 over EVERY registered entry, not just the pinned
    two: an over-wide band anywhere in the envelope is a finding. A drift_x2
    entry with a finding pytest.fail()s carrying drift, band, ceiling and
    provenance; a masked_ceiling entry must carry the finding it was
    registered on (re-emitted as a warning, so it stays visible) and is
    pytest.fail()ed when the numbers no longer justify it. A finding that
    is neither recorded nor visible is absorption by another name."""
    metrics = _envelope_metrics(bdd_context)
    for name, entry in sorted(metrics.items()):
        _surface_band_findings(name, entry)


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


# ---------------------------------------------------------------------------
# D1 — arch plumbing (@d1 @arch): metadata + every tensor created, no compute.
#
# "Loaded on CPU with compute disabled" = llama-server -ngl 0 --no-warmup with
# CUDA hidden and no request submitted: the loader reads the header, the arch
# creates and allocates every tensor, the context reserves its buffers, and no
# forward pass runs. llama-server is the one binary whose "no forward pass"
# claim is observable from outside the process: --metrics exposes the
# prompt_tokens_total / tokens_predicted_total counters (both must read 0) on
# top of the log (no "warming up" line, no timing lines). The server is
# started with LLAMA_TRACE=1 so the loader lists every tensor with its shape
# ("- tensor N, split S: name type [ ne0, ne1, ... ]"); those lines are
# cross-checked against the GGUF header read with gguf-py. Load success is
# itself the "every tensor created" proof (done_getting_tensors throws "wrong
# number of tensors" otherwise, and create_tensor throws on a shape mismatch);
# the trace listing turns it into a name-by-name, shape-by-shape assertion.
#
# Model under test: INKLING_MODEL_PATH when bound (Inkling-Small in a booked
# window), otherwise the tiny synthetic fixture generated in-step (CI policy
# in the feature header: this lane needs no oracle and no full model). The
# oracle directory is not a prerequisite here — D1 binds no expected value.
# ---------------------------------------------------------------------------

D1_N_CTX = 256
D1_THREADS = os.environ.get("INKLING_BDD_THREADS", "4")
D1_STARTUP_TIMEOUT_S = float(os.environ.get("INKLING_SERVER_STARTUP_TIMEOUT", "1800"))
D1_KV_PREFIX = "inkling."
D1_TENSOR_LINE = re.compile(
    r"- tensor\s+(\d+), split\s+(\d+):\s+(\S+)\s+(\S+)\s+\[\s*([0-9,\s]+?)\s*\]"
)
D1_KV_LINE = re.compile(r"- kv\s+(\d+):\s+(\S+)\s+(\S+)\s+=\s+(.*)$", re.MULTILINE)
D1_METRIC_LINE = re.compile(r"^llamacpp:(\w+)\s+([-+0-9.eE]+)\s*$", re.MULTILINE)
D1_FORWARD_PATTERNS = (
    r"warming up the model",
    r"prompt eval time",
    r"(?<!prompt )eval time\s*=",
    r"n_past = ",
    r"kv cache rm",
)


def _greedy_tool():
    """tools/inkling-greedy-dump.py as a module (its name is not importable)."""
    import importlib.util

    path = REPO_ROOT / "tools" / "inkling-greedy-dump.py"
    if not path.is_file():
        pytest.fail(f"{path} is missing; the server-driving helper lives there")
    spec = importlib.util.spec_from_file_location("inkling_greedy_dump", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_inkling_fixture(bdd_context, name: str = D6_FIXTURE_NAME) -> dict:
    """Run tools/make-inkling-test-gguf.py into tmp_path and read its header back.

    Shared by the D1/D4a/D4b lanes (the D6 Given wraps it). Returns the D6
    header dict ({"path", "size", "arch", "n_kv", "n_tensors",
    "tensor_names", ...}); a missing generator/numpy is reported by raising
    pytest.skip here because every caller treats it as an unmet prerequisite.
    """
    if not D6_GENERATOR.is_file():
        pytest.skip(f"Prerequisite unmet: D6 generator {D6_GENERATOR} is missing")
    try:
        import numpy  # noqa: F401
    except ImportError:
        pytest.skip("Prerequisite unmet: numpy is not importable (the fixture generator needs it)")
    out = bdd_context.tmp_path / name
    completed = subprocess.run(
        [sys.executable, str(D6_GENERATOR), str(out)],
        capture_output=True, text=True, timeout=D6_TIMEOUT_S,
        cwd=bdd_context.tmp_path, check=False,
    )
    if completed.returncode != 0 or not out.is_file():
        pytest.fail(
            f"fixture generator exited {completed.returncode} without writing {out}; "
            f"stderr tail:\n{completed.stderr[-2000:]}"
        )
    header = _d6_read_header(out)
    if header["arch"] != D6_ARCH:
        pytest.fail(f"fixture generator wrote general.architecture = {header['arch']!r}, expected {D6_ARCH!r}")
    return {**header, "path": out, "size": out.stat().st_size, "unmet": None}


def _gguf_header_full(path: Path) -> dict:
    """Header facts D1 cross-checks: every KV key, every tensor's name and ne shape."""
    if str(D6_GGUF_PY) not in sys.path:
        sys.path.insert(0, str(D6_GGUF_PY))
    from gguf import GGUFReader

    reader = GGUFReader(str(path))
    return {
        "arch": _gguf_string(reader.fields["general.architecture"]),
        "n_kv": _gguf_scalar(reader.fields["GGUF.kv_count"]),
        "n_tensors": _gguf_scalar(reader.fields["GGUF.tensor_count"]),
        "kv_keys": [name for name in reader.fields if not name.startswith("GGUF.")],
        # ReaderTensor.shape is the file's dims array = ggml ne order (ne0 first)
        "tensor_shapes": {tensor.name: [int(dim) for dim in tensor.shape] for tensor in reader.tensors},
    }


def _d1_model(bdd_context) -> tuple[Path, str]:
    """(model path, provenance): INKLING_MODEL_PATH when bound, else the fixture."""
    bound = getattr(bdd_context, "inkling_model_path", None)
    if bound is not None:
        return Path(bound), "INKLING_MODEL_PATH"
    fixture = generate_inkling_fixture(bdd_context, "inkling-d1.gguf")
    return fixture["path"], "synthetic fixture (tools/make-inkling-test-gguf.py)"


@given(parsers.parse("a dreamcatcher build with LLM_ARCH_INKLING registered"))
def given_d1_build(bdd_context):
    """Resolve llama-server (LLAMA_SERVER_BIN, build/bin, PATH); recording only."""
    path, unmet = _find_binary("llama-server", "LLAMA_SERVER_BIN")
    bdd_context.inkling_d1 = {"server": path, "unmet": unmet}


@when(parsers.parse("the model is loaded on CPU with compute disabled"))
def when_d1_load(bdd_context):
    state = getattr(bdd_context, "inkling_d1", None)
    if state is None:
        pytest.skip("Prerequisite unmet: the D1 build Given did not run")
    if state["unmet"]:
        pytest.skip(f"Prerequisite unmet: {state['unmet']}")
    tool = _greedy_tool()
    model, provenance = _d1_model(bdd_context)
    header = _gguf_header_full(model)
    if header["arch"] != D6_ARCH:
        pytest.fail(f"{model}: general.architecture is {header['arch']!r}; the D1 lane needs an inkling GGUF")

    port = tool.free_port()
    log_path = bdd_context.tmp_path / "d1-llama-server.log"
    bdd_context.inkling_launches = getattr(bdd_context, "inkling_launches", 0) + 1
    process = tool.start_server(
        state["server"],
        ["-m", str(model)],
        port,
        log_path,
        f"-ngl 0 -t {D1_THREADS} -c {D1_N_CTX} --no-warmup --metrics "
        + os.environ.get("INKLING_D1_SERVER_ARGS", ""),
        env={"LLAMA_TRACE": "1"},
    )
    bdd_context.processes.append(process)
    try:
        tool.wait_health(process, port, D1_STARTUP_TIMEOUT_S)
        status, metrics = tool.http_text(f"http://127.0.0.1:{port}/metrics", timeout=30.0)
        if status != 200:
            pytest.fail(f"/metrics answered {status} (--metrics not honoured?): {metrics[:300]}")
        _, props = tool.http_json(f"http://127.0.0.1:{port}/props", timeout=30.0)
    except RuntimeError as error:
        tail = log_path.read_bytes()[-3000:].decode("utf-8", "replace") if log_path.is_file() else ""
        pytest.fail(f"D1 load: {error}\n--- server log tail ---\n{tail}")
    finally:
        tool.stop_server(process)
    bdd_context.inkling_d1_run = {
        "model": model,
        "provenance": provenance,
        "header": header,
        "log": log_path.read_bytes().decode("utf-8", "replace"),
        "metrics": {name: float(value) for name, value in D1_METRIC_LINE.findall(metrics)},
        "props": props if isinstance(props, dict) else {},
    }


def _d1_run_or_fail(bdd_context) -> dict:
    run = getattr(bdd_context, "inkling_d1_run", None)
    if not run:
        pytest.fail("no D1 load recorded: the When must load the model before this gate (fail closed)")
    return run


@then(parsers.parse("every tensor is created with the Inkling names and shapes"))
def then_d1_tensors(bdd_context):
    run = _d1_run_or_fail(bdd_context)
    log, header = run["log"], run["header"]
    failures: list[str] = []

    loaded = re.search(r"loaded meta data with (\d+) key-value pairs and (\d+) tensors", log)
    if loaded is None:
        failures.append("loader never reported 'loaded meta data with ... tensors'")
    elif (int(loaded.group(1)), int(loaded.group(2))) != (header["n_kv"], header["n_tensors"]):
        failures.append(
            f"loader read {loaded.group(1)} key-value pairs / {loaded.group(2)} tensors, header has "
            f"{header['n_kv']} / {header['n_tensors']}"
        )
    if re.search(rf"arch\s*=\s*{D6_ARCH}\b", log) is None:
        failures.append(f"print-meta never reported 'arch = {D6_ARCH}' (LLM_ARCH_INKLING not resolved)")
    if re.search(r"llm_load_tensors:.*buffer size\s*=\s*[0-9.]+ MiB", log) is None:
        failures.append("llm_load_tensors never reported a weight buffer (tensors were not created/allocated)")
    if error_line := _d6_first_match(D6_LOAD_ERROR_PATTERNS, log):
        failures.append(f"loader error: {error_line}")

    # name-by-name, shape-by-shape: the loader's trace listing vs the GGUF header
    listed = {
        name: [int(dim) for dim in shape.replace(" ", "").split(",") if dim]
        for _, _, name, _, shape in D1_TENSOR_LINE.findall(log)
    }
    expected = header["tensor_shapes"]
    if not listed:
        failures.append("loader printed no '- tensor N, split S: ...' lines (LLAMA_TRACE=1 not honoured?)")
    else:
        if set(listed) != set(expected):
            missing = sorted(set(expected) - set(listed))[:8]
            extra = sorted(set(listed) - set(expected))[:8]
            failures.append(f"tensor name set differs from the header: missing {missing}, unexpected {extra}")
        for name, shape in sorted(listed.items()):
            want = expected.get(name)
            if want is not None and shape != want[: len(shape)] + [1] * (len(shape) - len(want)) and shape != want:
                failures.append(f"{name}: loader shape {shape} != header ne {want}")
        if len(listed) != header["n_tensors"]:
            failures.append(f"loader listed {len(listed)} tensors, header has {header['n_tensors']}")
    if failures:
        pytest.fail("D1 tensor creation failed:\n" + "\n".join(f"  - {line}" for line in failures))
    _record_diagnostic(
        f"D1: {len(listed)} Inkling tensors created with header names/shapes on {run['provenance']}"
    )


@then(parsers.parse("the KV keys include {keys}"))
def then_d1_kv_keys(bdd_context, keys: str):
    """Every listed key (inkling.<key>) is in the file AND was read by the loader."""
    run = _d1_run_or_fail(bdd_context)
    wanted = [D1_KV_PREFIX + key.strip() for key in re.split(r",\s*(?:and\s+)?|\s+and\s+", keys) if key.strip()]
    if not wanted:
        pytest.fail("the scenario names no KV keys")
    in_file = set(run["header"]["kv_keys"])
    in_loader = {name for _, name, _, _ in D1_KV_LINE.findall(run["log"])}
    if not in_loader:
        pytest.fail("loader printed no '- kv N: ...' lines; the header was never enumerated")
    missing_file = [key for key in wanted if key not in in_file]
    missing_loader = [key for key in wanted if key not in in_loader]
    if missing_file or missing_loader:
        pytest.fail(
            f"KV keys missing from the GGUF header: {missing_file}; "
            f"missing from the loader's enumeration: {missing_loader}"
        )


@then(parsers.parse("no forward pass has run"))
def then_d1_no_forward(bdd_context):
    run = _d1_run_or_fail(bdd_context)
    metrics = run["metrics"]
    failures: list[str] = []
    for counter in ("prompt_tokens_total", "tokens_predicted_total"):
        if counter not in metrics:
            failures.append(f"/metrics carries no {counter} counter: {sorted(metrics)}")
        elif metrics[counter] != 0:
            failures.append(f"/metrics {counter} = {metrics[counter]:g}, a forward pass ran")
    if line := _d6_first_match(D1_FORWARD_PATTERNS, run["log"]):
        failures.append(f"the server log shows compute: {line}")
    if failures:
        pytest.fail("D1 'no compute' violated:\n" + "\n".join(f"  - {line}" for line in failures))


# ---------------------------------------------------------------------------
# D3 — hybrid attention + recurrent cache (@d3 @cache). Model-heavy: the gate
# is the KLD / top-1 envelope against the banded D0 oracle (feature header,
# D3 gate 2026-09-19), so the lane runs Inkling-Small in a booked window.
#
# Given: the model INKLING_MODEL_PATH names must BE the hybrid 5:1 iswa model
# (header: sliding_window > 0, sliding_window_pattern with five SWA layers per
# global layer, shortconv_kernel > 1 for the recurrent short-conv state). The
# synthetic fixture is deliberately NOT a stand-in — its pattern is 1:1 and
# there is no oracle for it — so an unbound model is an unmet prerequisite.
# When: INKLING_D3_PPL_CMD, the same {model} {text} {base} template as
# INKLING_PPL_CMD but running the D3 cache path (-fa on: one cache +
# GGML_OP_FLASH_ATTN_EXT_BANDED, cf. build_inkling.cpp), over the D0 4x2048
# split: every 2048-token chunk crosses n_ctx / sliding_window - 1 window
# boundaries (3 at the real model's 512), which is the "several" the scenario
# asks for. "int64 expert indexing" is the build under test's routing
# (int64 ids into the expert bank); this harness cannot see inside the graph,
# so it records the build's declared features and gates on the observable
# consequence below.
# Then (consistency): the run reached the end of every chunk with a finite
# perplexity and a finite KLD / top-1 summary — a cache that loses attention
# state, short-conv state or expert indices across a window boundary shows up
# as NaN/Inf or a collapsed agreement — and the context confirms the FA cache
# path (flash_attn = 1). The envelope Then (shared with D2) is the gate.
# ---------------------------------------------------------------------------

D3_SWA_RATIO = (5, 1)


def _inkling_swa_header(model: Path) -> dict:
    if str(D6_GGUF_PY) not in sys.path:
        sys.path.insert(0, str(D6_GGUF_PY))
    from gguf import GGUFReader

    reader = GGUFReader(str(model))

    def field(name: str):
        entry = reader.fields.get(name)
        if entry is None:
            pytest.fail(f"{model}: GGUF header has no {name}")
        return entry.contents()

    if field("general.architecture") != D6_ARCH:
        pytest.fail(f"{model}: general.architecture is not {D6_ARCH!r}")
    pattern = [int(value) for value in field("inkling.attention.sliding_window_pattern")]
    return {
        "n_layer": int(field("inkling.block_count")),
        "n_swa": int(field("inkling.attention.sliding_window")),
        "pattern": pattern,
        "shortconv_kernel": int(field("inkling.shortconv_kernel")),
        "expert_count": int(field("inkling.expert_count")),
    }


@given(parsers.parse("a hybrid Inkling cache with iswa windows over a 5:1 pattern"))
def given_d3_cache(bdd_context):
    state: dict = {"unmet": None, "header": None}
    model = getattr(bdd_context, "inkling_model_path", None)
    if model is None:
        state["unmet"] = (
            "INKLING_MODEL_PATH is not bound to a readable file (the @d3 lane runs Inkling-Small "
            "in a booked window; the synthetic fixture has a 1:1 window pattern and no oracle)"
        )
    else:
        header = _inkling_swa_header(Path(model))
        swa, total = sum(1 for value in header["pattern"] if value), len(header["pattern"])
        n_global = total - swa
        want_swa, want_global = D3_SWA_RATIO
        if header["n_swa"] <= 0 or total == 0 or n_global == 0 or swa * want_global != n_global * want_swa:
            pytest.fail(
                f"{model}: not the hybrid {want_swa}:{want_global} iswa model — sliding_window "
                f"{header['n_swa']}, pattern has {swa} SWA / {n_global} global layers"
            )
        if header["shortconv_kernel"] <= 1:
            pytest.fail(f"{model}: shortconv_kernel {header['shortconv_kernel']} carries no recurrent state")
        state["header"] = header
    bdd_context.inkling_d3 = state


@when(parsers.parse("generation runs past several window boundaries with int64 expert indexing"))
def when_d3_run(bdd_context):
    state = getattr(bdd_context, "inkling_d3", None)
    if state is None:
        pytest.skip("Prerequisite unmet: the D3 cache Given did not run")
    if state["unmet"]:
        pytest.skip(f"Prerequisite unmet: {state['unmet']}")
    directory, model, command = _require(bdd_context, "INKLING_D3_PPL_CMD")
    text = os.environ.get("INKLING_PPL_TEXT")
    if not text or not Path(text).is_file():
        pytest.skip("Prerequisite unmet: INKLING_PPL_TEXT is not bound to a readable file")
    base = directory / "kld-base-4x2048.bin"
    stdout, stderr = _run_command_full(command, model=str(model), text=text, base=str(base))
    log = stderr + "\n" + stdout
    rms = re.search(r"RMS\s+(?:Δp|dp)\s*:\s*([-+0-9.eE]+|nan|inf)", log)
    top1 = re.search(r"Same top p:\s*([-+0-9.eE]+|nan|inf)", log)
    if rms is None or top1 is None:
        pytest.fail(
            'D3 run did not yield the KL-divergence summary lines "RMS Δp    : X ± Y %" and '
            f'"Same top p: Z ± W %" (RMS line: {"yes" if rms else "no"}, top-1 line: '
            f'{"yes" if top1 else "no"}); INKLING_D3_PPL_CMD must run --kl-divergence against {{base}}'
        )
    estimate = re.search(r"Final estimate: PPL over (\d+) chunks for n_ctx=(\d+)", log)
    chunks = [float(value) for value in re.findall(r"\[\d+\]([-+0-9.eE]+|nan|inf),", log)]
    if not chunks:
        # This fork's --kl-divergence mode prints neither the mainline "[n]X," chunks nor a
        # "Final estimate: PPL over N chunks for n_ctx=M" summary; it prints a per-chunk table
        # (chunk | PPL | ln(PPL(Q)/PPL(base)) | KL Divergence | Δp RMS | Same top p). Read the
        # PPL column: the first value before the "±" on each numbered row.
        chunks = [
            float(value)
            for value in re.findall(r"(?m)^\s*\d+\s+([0-9.]+(?:[eE][-+]?\d+)?)\s+±", log)
        ]
    if estimate:
        n_ctx = int(estimate.group(2))
        n_chunks = int(estimate.group(1))
    else:
        ctx_match = re.search(r"n_ctx\s+=\s*(\d+)", log)
        n_ctx = int(ctx_match.group(1)) if ctx_match else 0
        n_chunks = len(chunks)
    n_swa = state["header"]["n_swa"]
    bdd_context.inkling_observed = {
        "top1_disagreement": 100.0 - float(top1.group(1)),
        "rms_dp": float(rms.group(1)),
    }
    bdd_context.inkling_d3_run = {
        "log": log,
        "chunks": chunks,
        "n_chunks": n_chunks,
        "n_ctx": n_ctx,
        "boundaries_per_chunk": (n_ctx // n_swa - 1) if n_swa > 0 and n_ctx > 0 else 0,
        "flash_attn": re.search(r"flash_attn\s*=\s*1\b", log) is not None,
        "features": _declared_features("INKLING_BUILD_FEATURES"),
    }


@then(parsers.parse("attention state, recurrent short-conv state, and expert indices stay consistent"))
def then_d3_consistent(bdd_context):
    run = getattr(bdd_context, "inkling_d3_run", None)
    if not run:
        pytest.fail("no D3 run recorded: the When must run the cache path before this gate (fail closed)")
    failures: list[str] = []
    if run["n_chunks"] < 1 or not run["chunks"]:
        failures.append("no per-chunk perplexity reached the summary: the run did not complete a chunk")
    elif len(run["chunks"]) < run["n_chunks"]:
        failures.append(f"only {len(run['chunks'])} of {run['n_chunks']} chunk perplexities were printed")
    for index, value in enumerate(run["chunks"], 1):
        if not math.isfinite(value) or value <= 0:
            failures.append(f"chunk {index} perplexity {value} is not finite: state went inconsistent")
    for name, value in sorted(getattr(bdd_context, "inkling_observed", {}).items()):
        if not math.isfinite(value):
            failures.append(f"{name} = {value}: the KL-divergence summary is not finite")
    if run["boundaries_per_chunk"] < 2:
        failures.append(
            f"each chunk crosses {run['boundaries_per_chunk']} window boundaries (n_ctx {run['n_ctx']}); "
            f"'several' needs at least 2 — raise the run's n_ctx"
        )
    if not run["flash_attn"]:
        failures.append("the context did not report flash_attn = 1: INKLING_D3_PPL_CMD is not running the banded cache path (-fa on)")
    if line := _d6_first_match(D6_LOAD_ERROR_PATTERNS, run["log"]):
        failures.append(f"run error: {line}")
    if failures:
        pytest.fail("D3 cache consistency failed:\n" + "\n".join(f"  - {line}" for line in failures))
    _record_diagnostic(
        f"D3: {run['n_chunks']} chunks x {run['boundaries_per_chunk']} window boundaries, per-chunk PPL "
        f"{run['chunks']}, banded cache path; build features "
        f"{sorted(run['features']) if run['features'] else 'undeclared (INKLING_BUILD_FEATURES)'}"
    )
