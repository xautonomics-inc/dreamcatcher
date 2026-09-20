#!/usr/bin/env python3
"""Drive a fixed prompt set through llama-server greedily and print the dump in the D0 schema.

The D0 oracle (greedy-64x8.json, feature header of tests/bdd/features/inkling.feature)
was recorded by driving llama-server's /completion endpoint with

    {"prompt": P, "n_predict": 64, "temperature": 0, "top_k": 1, "seed": 1,
     "cache_prompt": false, "return_tokens": true, "n_probs": 10}

and keeping, per prompt, {"i", "prompt", "tokens", "content",
"completion_probabilities", "timings"} where completion_probabilities is the
server's per-position list of {"id", "token", "bytes", "logprob",
"top_logprobs": [{"id", "token", "bytes", "logprob"}, ...]}. This runner produces
exactly that shape from any llama-server binary, so the @d2 greedy scenario
(INKLING_GREEDY_CMD) and the @d4a library scenario compare like with like.

The ik-lineage server answers the same request in its legacy shape — "tokens" =
the generated ids, completion_probabilities = [{"content": piece, "probs":
[{"tok_str", "prob"}, ...]}] — which normalize_record() lifts into the D0 shape:
id from "tokens", logprob = log(prob) of the sampled token (matched by piece in
its own top-n list, which holds it for a greedy pick), top_logprobs =
log(prob) of every listed candidate. A dump that already carries "id"/"logprob"
(a mainline-lineage server, the oracle itself) is passed through unchanged.

Usage (shell-free, one process; the BDD steps call it with {model}/{oracle_dir}
placeholders):

    inkling-greedy-dump.py --server BIN (--model GGUF | --model-dir LIB) \
        --prompts FILE [--n-predict 64] [--n-probs 10] [--port N] \
        [--server-args "-ngl 0 -t 20 -c 4096 -np 1 -fa off"] \
        [--startup-timeout S] [--request-timeout S] [--out FILE] [--server-log FILE]

--prompts is the oracle dump itself (a list of records carrying "prompt"), a
{"prompts": [...]} object, or a plain JSON list of strings. The dump goes to
stdout (and --out when given); everything else goes to stderr. CUDA is hidden
(CUDA_VISIBLE_DEVICES="") so a CUDA-built server runs CPU-only like the oracle.
The server is always stopped on exit, including on failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

D0_REQUEST = {
    "n_predict": 64,
    "temperature": 0,
    "top_k": 1,
    "seed": 1,
    "cache_prompt": False,
    "return_tokens": True,
    "n_probs": 10,
}


def free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def sha256_of(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prompts(path: str | os.PathLike[str]) -> list[str]:
    data = json.loads(Path(path).read_bytes())
    if isinstance(data, dict):
        data = data.get("prompts", data)
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path}: expected a non-empty list of prompts or prompt records")
    prompts: list[str] = []
    for index, entry in enumerate(data):
        if isinstance(entry, str):
            prompts.append(entry)
        elif isinstance(entry, dict) and isinstance(entry.get("prompt"), str):
            prompts.append(entry["prompt"])
        else:
            raise ValueError(f"{path}: record {index} carries no \"prompt\" string")
    return prompts


def start_server(
    server_bin: str,
    model_args: list[str],
    port: int,
    log_path: str | os.PathLike[str],
    extra_args: str = "",
    env: dict[str, str] | None = None,
) -> subprocess.Popen[bytes]:
    """Launch llama-server CPU-only; stdout+stderr go to log_path."""
    argv = [server_bin, *model_args, "--host", "127.0.0.1", "--port", str(port), *shlex.split(extra_args)]
    environment = {**os.environ, **(env or {}), "CUDA_VISIBLE_DEVICES": ""}
    log_handle = open(log_path, "wb")
    try:
        process = subprocess.Popen(
            argv, stdout=log_handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=environment
        )
    finally:
        log_handle.close()
    return process


def stop_server(process: subprocess.Popen[bytes] | None, grace: float = 10.0) -> None:
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def http_json(url: str, body: dict | None = None, timeout: float = 30.0) -> tuple[int, object]:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
            return response.status, (json.loads(payload) if payload else None)
    except urllib.error.HTTPError as error:
        payload = error.read()
        try:
            return error.code, json.loads(payload)
        except ValueError:
            return error.code, payload.decode("utf-8", "replace")


def http_text(url: str, timeout: float = 30.0) -> tuple[int, str]:
    request = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", "replace")


def wait_health(process: subprocess.Popen[bytes], port: int, timeout: float) -> None:
    """Block until /health answers 200 or the server dies / the timeout passes."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited {process.returncode} during startup")
        try:
            status, _ = http_json(f"http://127.0.0.1:{port}/health", timeout=5.0)
        except (urllib.error.URLError, OSError, ValueError):
            status = 0
        if status == 200:
            return
        time.sleep(0.2)
    raise RuntimeError(f"llama-server did not become healthy on port {port} within {timeout:.0f} s")


def _logprob(prob: float) -> float:
    return math.log(prob) if prob > 0 else -1e30  # a masked candidate (p = 0) stays JSON-encodable


def normalize_record(response: dict, index: int) -> tuple[list[int], list[dict]]:
    """(token ids, D0-shape completion_probabilities) from either server lineage."""
    probabilities = response.get("completion_probabilities")
    if not isinstance(probabilities, list) or not probabilities:
        raise RuntimeError(
            f"prompt {index}: response carries no completion_probabilities (n_probs requested); "
            f"keys: {sorted(response)}"
        )
    tokens = response.get("tokens")
    if probabilities and isinstance(probabilities[0], dict) and "id" in probabilities[0]:
        # mainline shape (the oracle's): pass through, ids from the entries when not echoed
        ids = [int(token) for token in tokens] if tokens else [int(entry["id"]) for entry in probabilities]
        return ids, probabilities
    # ik legacy shape: {"content": piece, "probs": [{"tok_str", "prob"}, ...]}
    if not tokens:
        raise RuntimeError(f"prompt {index}: legacy-shape response without \"tokens\"; ids are unrecoverable")
    ids = [int(token) for token in tokens]
    if len(ids) != len(probabilities):
        raise RuntimeError(
            f"prompt {index}: {len(ids)} generated ids but {len(probabilities)} probability entries"
        )
    normalized = []
    for position, (token, entry) in enumerate(zip(ids, probabilities)):
        piece = entry.get("content")
        candidates = entry.get("probs") or []
        match = next((cand for cand in candidates if cand.get("tok_str") == piece), None)
        if match is None:
            raise RuntimeError(
                f"prompt {index} position {position}: the sampled piece {piece!r} is not in its top-"
                f"{len(candidates)} list; a greedy pick is always its own top-1 (with a grammar, "
                f"raise n_probs — the list is the raw top-n)"
            )
        normalized.append({
            "id": token,
            "token": piece,
            "logprob": _logprob(float(match["prob"])),
            "top_logprobs": [
                {"token": cand.get("tok_str"), "logprob": _logprob(float(cand["prob"]))} for cand in candidates
            ],
        })
    return ids, normalized


def greedy_record(port: int, index: int, prompt: str, n_predict: int, n_probs: int, timeout: float,
                  extra: dict | None = None) -> dict:
    """One prompt through /completion with the D0 body (+ extra fields, e.g. a grammar)."""
    body = {**D0_REQUEST, "prompt": prompt, "n_predict": n_predict, "n_probs": n_probs, **(extra or {})}
    status, response = http_json(f"http://127.0.0.1:{port}/completion", body, timeout=timeout)
    if status != 200 or not isinstance(response, dict):
        raise RuntimeError(f"prompt {index}: /completion answered {status}: {str(response)[:500]}")
    tokens, probabilities = normalize_record(response, index)
    return {
        "i": index,
        "prompt": prompt,
        "tokens": tokens,
        "content": response.get("content"),
        "completion_probabilities": probabilities,
        "timings": response.get("timings"),
    }


def greedy_dump(port: int, prompts: list[str], n_predict: int, n_probs: int, timeout: float,
                extra: dict | None = None) -> list[dict]:
    return [
        greedy_record(port, index, prompt, n_predict, n_probs, timeout, extra)
        for index, prompt in enumerate(prompts)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", required=True, help="llama-server binary")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="monolithic GGUF (-m)")
    source.add_argument("--model-dir", help="per-layer library directory (--model-dir)")
    parser.add_argument("--layers", help="with --model-dir: absolute block window A,B")
    parser.add_argument("--prompts", required=True, help="oracle dump / prompt list (JSON)")
    parser.add_argument("--n-predict", type=int, default=D0_REQUEST["n_predict"])
    parser.add_argument("--n-probs", type=int, default=D0_REQUEST["n_probs"])
    parser.add_argument("--port", type=int, default=0, help="0 = pick a free port")
    parser.add_argument("--server-args", default="", help="extra llama-server arguments (one string)")
    parser.add_argument("--startup-timeout", type=float, default=1800.0, help="seconds to wait for /health")
    parser.add_argument("--request-timeout", type=float, default=1800.0, help="seconds per /completion")
    parser.add_argument("--grammar", help="GBNF grammar added to every request (fixture runs: "
                        "'root ::= [ -~]*' keeps a byte-level vocab inside JSON-safe ASCII); never for an oracle run")
    parser.add_argument("--out", help="also write the dump here")
    parser.add_argument("--server-log", help="server stdout/stderr (default: <out>.server.log or a temp file)")
    args = parser.parse_args(argv)

    prompts = load_prompts(args.prompts)
    model_args = ["-m", args.model] if args.model else ["--model-dir", args.model_dir]
    if args.layers:
        if not args.model_dir:
            parser.error("--layers needs --model-dir")
        model_args += ["--layers", args.layers]
    port = args.port or free_port()
    if args.server_log:
        log_path = Path(args.server_log)
    elif args.out:
        log_path = Path(args.out + ".server.log")
    else:
        log_path = Path(os.environ.get("TMPDIR", "/tmp")) / f"inkling-greedy-dump-{os.getpid()}.server.log"

    print(
        f"inkling-greedy-dump: server={args.server} sha256={sha256_of(args.server)[:16]} "
        f"model_args={model_args} port={port} prompts={len(prompts)} n_predict={args.n_predict} "
        f"n_probs={args.n_probs} log={log_path}",
        file=sys.stderr,
        flush=True,
    )
    process = None
    try:
        process = start_server(args.server, model_args, port, log_path, args.server_args)
        wait_health(process, port, args.startup_timeout)
        # a grammar masks the argmax; the ik server lists the RAW top-n, so ask for a
        # --n-probs large enough (the vocabulary) to keep the sampled token in the list
        extra = {"grammar": args.grammar} if args.grammar else None
        records = greedy_dump(port, prompts, args.n_predict, args.n_probs, args.request_timeout, extra)
    except Exception as error:  # noqa: BLE001 - report and fail closed
        print(f"inkling-greedy-dump: FAILED: {error}", file=sys.stderr)
        try:
            tail = log_path.read_bytes()[-3000:].decode("utf-8", "replace")
            print(f"--- server log tail ---\n{tail}", file=sys.stderr)
        except OSError:
            pass
        return 1
    finally:
        stop_server(process)

    payload = json.dumps(records, indent=1)
    if args.out:
        Path(args.out).write_text(payload)
    sys.stdout.write(payload)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
