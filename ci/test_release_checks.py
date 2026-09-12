"""Executable fixtures and local HTTP protocol checks for release gates."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import check_image_entrypoints as entrypoints
import smoke_serve


class Handler(BaseHTTPRequestHandler):
    mode = "ok"
    requests: list[str] = []
    payload: object = None

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        self.requests.append(self.path)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"data":[{"id":"fixture-model"}]}')

    def do_POST(self) -> None:
        self.requests.append(self.path)
        Handler.payload = json.loads(
            self.rfile.read(int(self.headers["Content-Length"]))
        )
        if self.mode == "http_error":
            self.send_response(503)
            self.end_headers()
            self.wfile.write(b"private diagnostic must not escape")
            return
        if self.mode == "redirect":
            self.send_response(307)
            self.send_header("Location", "/unexpected")
            self.end_headers()
            return
        self.send_response(200)
        self.end_headers()
        if self.mode == "invalid_json":
            self.wfile.write(b"not json")
            return
        content = "" if self.mode in ("empty", "reasoning") else "A bird flew."
        tokens: object = 0 if self.mode == "zero" else 8
        if self.mode == "bool_tokens":
            tokens = True
        reply: dict[str, object] = {
            "choices": [
                {
                    "message": {
                        "content": content,
                        "reasoning_content": "Thinking"
                        if self.mode == "reasoning"
                        else "",
                    }
                }
            ],
            "usage": {"completion_tokens": tokens},
        }
        if self.mode == "missing_usage":
            del reply["usage"]
        self.wfile.write(json.dumps(reply).encode())


class SmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        Handler.mode = "ok"
        Handler.requests = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_real_post_and_usage(self) -> None:
        result = smoke_serve.smoke(self.base, 0, "", "", 2)
        self.assertTrue(result["ok"])
        self.assertEqual(result["completion_tokens"], 8)
        self.assertEqual(Handler.requests, ["/v1/models", "/v1/chat/completions"])
        self.assertEqual(
            smoke_serve.obj(Handler.payload)["chat_template_kwargs"],
            {"enable_thinking": False},
        )
        self.assertIn("started_at", result)
        self.assertIn("completed_at", result)
        self.assertNotIn("A bird flew", json.dumps(result))

    def test_threshold_failure_and_explicit_model(self) -> None:
        result = smoke_serve.smoke(self.base + "/v1", 1e12, "fixture", "", 2)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "below_min_tps")
        self.assertEqual(Handler.requests, ["/v1/chat/completions"])

    def test_refusals_and_no_redirect(self) -> None:
        for mode in [
            "empty",
            "reasoning",
            "zero",
            "bool_tokens",
            "missing_usage",
            "invalid_json",
            "http_error",
            "redirect",
        ]:
            with self.subTest(mode=mode):
                Handler.mode = mode
                result = smoke_serve.smoke(self.base, 0, "fixture", "test-value", 2)
                self.assertFalse(result["ok"])
                if mode == "reasoning":
                    self.assertEqual(result["error"], "reasoning_only")
                    with (
                        patch("sys.argv", ["smoke", self.base]),
                        patch("builtins.print"),
                    ):
                        self.assertEqual(smoke_serve.main(), 1)
                self.assertNotIn("test-value", json.dumps(result))
                self.assertNotIn("private diagnostic", json.dumps(result))
        self.assertNotIn("/unexpected", Handler.requests)

    def test_bad_inputs_do_not_send(self) -> None:
        for rate in [-1, float("nan"), float("inf")]:
            self.assertFalse(smoke_serve.smoke(self.base, rate, "fixture", "", 2)["ok"])
        self.assertFalse(
            smoke_serve.smoke("http://name:value@example.org", 0, "fixture", "", 2)[
                "ok"
            ]
        )
        self.assertFalse(smoke_serve.smoke(self.base, 0, "fixture", "", 0)["ok"])
        self.assertEqual(Handler.requests, [])

    def test_timeout_and_cli_exit(self) -> None:
        with patch.object(smoke_serve, "request", side_effect=TimeoutError):
            self.assertFalse(smoke_serve.smoke(self.base, 0, "fixture", "", 1)["ok"])
        with (
            patch("sys.argv", ["smoke", self.base, "--min-tps", "1e12"]),
            patch("builtins.print"),
        ):
            self.assertEqual(smoke_serve.main(), 1)


class EntrypointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "CMakeLists.txt").write_text(
            "set(TARGET llama-server)\nadd_executable(${TARGET} server.cpp)\n"
        )
        self.container = self.root / "Containerfile"

    def run_check(self, text: str, target: str | None = None) -> dict[str, object]:
        self.container.write_text(text)
        return entrypoints.check(self.root, [self.container], target)

    def test_copy_and_entrypoint(self) -> None:
        result = self.run_check(
            "FROM scratch AS server\n"
            "COPY --from=build /build/bin/llama-server /app/\n"
            'ENTRYPOINT ["/app/llama-server"]\n'
        )
        self.assertTrue(result["ok"])

    def test_missing_target_and_copy_are_failures(self) -> None:
        for text in [
            'FROM scratch\nENTRYPOINT ["/app/llama-stage-runner"]\n',
            (
                "FROM scratch\n"
                'COPY --from=build ["/build/bin/llama-missing", "/app/"]\n'
                'ENTRYPOINT ["/app/llama-server"]\n'
            ),
            'FROM scratch\nENTRYPOINT ["/app/${BINARY}"]\n',
            "FROM scratch\nENTRYPOINT exec /app/llama-server\n",
        ]:
            with self.subTest(text=text):
                self.assertFalse(self.run_check(text)["ok"])

    def test_selected_stage_and_inherited_copy(self) -> None:
        text = (
            "FROM scratch AS base\n"
            "COPY --from=build /build/bin/llama-server /app/\n"
            "FROM base AS server\n"
            'ENTRYPOINT ["/app/llama-server"]\n'
            "FROM server AS swap\n"
            'ENTRYPOINT ["/app/llama-swap"]\n'
        )
        self.assertTrue(self.run_check(text, "server")["ok"])
        self.assertFalse(self.run_check(text)["ok"])
        self.assertFalse(self.run_check(text, "missing")["ok"])

    def test_script_and_comments(self) -> None:
        (self.root / "entry.sh").write_text('#!/bin/sh\nexec "$@"\n')
        self.assertTrue(self.run_check('FROM scratch\nENTRYPOINT ["/entry.sh"]')["ok"])
        (self.root / "CMakeLists.txt").write_text(
            "# add_executable(fake ignored.cpp)\n"
            "#[[\n"
            "add_executable(fake ignored.cpp)\n"
            "]]\n"
            'message("add_executable(fake ignored.cpp)")\n'
        )
        self.assertFalse(self.run_check('FROM scratch\nENTRYPOINT ["/fake"]')["ok"])

    def test_multiline_and_literal_target(self) -> None:
        (self.root / "CMakeLists.txt").write_text(
            "add_executable(\n llama-server\n server.cpp\n)"
        )
        self.assertTrue(
            self.run_check(
                "FROM scratch\n"
                "COPY --from=build \\\n"
                " /build/bin/llama-server /app/\n"
                'ENTRYPOINT ["/app/llama-server"]'
            )["ok"]
        )

    def test_cli_exit(self) -> None:
        self.container.write_text('FROM scratch\nENTRYPOINT ["/missing"]')
        with (
            patch("sys.argv", ["check", "--root", str(self.root), str(self.container)]),
            patch("builtins.print"),
        ):
            self.assertEqual(entrypoints.main(), 1)


if __name__ == "__main__":
    unittest.main()
