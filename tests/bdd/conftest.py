"""BDD test configuration and fixtures for Dreamcatcher scenarios."""

from __future__ import annotations

import os
import re
import sys
import time
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Any, Generator, Dict

import pytest
import requests

# Add repo root and tools to sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LAYER_DIST_PATH = REPO_ROOT / "tools" / "layer-distribution"
if str(LAYER_DIST_PATH) not in sys.path:
    sys.path.insert(0, str(LAYER_DIST_PATH))


def pytest_bdd_apply_tag(tag: str, function: Any) -> bool | None:
    """Treat @known-issue tags as expected failures (xfail)."""
    if tag == "known-issue":
        marker = pytest.mark.xfail(reason="Known issue tracked upstream in meta tracker", strict=False)
        marker(function)
        return True
    return None


class BddContext:
    """Holds runtime state across steps in a single scenario."""

    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path
        self.processes: list[subprocess.Popen] = []
        self.lib_dir: Path | None = None
        self.monolith_path: Path | None = None
        self.host: str = os.environ.get("BDD_HOST", "127.0.0.1")
        self.port: int = int(os.environ.get("BDD_PORT", "8080"))
        self.last_response: requests.Response | None = None
        self.last_proc: subprocess.Popen | None = None
        self.last_returncode: int | None = None
        self.last_stdout: str = ""
        self.last_stderr: str = ""
        self.manifest: dict[str, Any] | None = None
        self.files_for_window_res: set[str] = set()
        self.precheck_res: Any = None
        self.rebalance_res: Any = None
        self.stages: dict[str, subprocess.Popen] = {}
        self.browser_page: Any = None
        self.server_started: bool = False
        self.server_url: str = ""
        self.part_hashes: dict[str, str] = {}
        self.req_bytes: int = 0
        self.window_str: str = ""
        self.old_window: tuple[int, int] = (0, 0)
        self.new_window: tuple[int, int] = (0, 0)

    def resolve_placeholder(self, val: str) -> str:
        """Resolve placeholders like <lib_dir>, <host>, <port>, <monolith_path>."""
        if not isinstance(val, str):
            return val
        if val == "<lib_dir>":
            if not self.lib_dir:
                self.lib_dir = self.tmp_path / "model_lib"
                self.lib_dir.mkdir(parents=True, exist_ok=True)
            return str(self.lib_dir)
        if val == "<monolith_path>":
            if not self.monolith_path:
                self.monolith_path = self.tmp_path / "model.gguf"
            return str(self.monolith_path)
        if val == "<host>":
            return self.host
        if val == "<port>":
            return str(self.port)
        if val == "<corrupt_dir>":
            corrupt = self.tmp_path / "corrupt_lib"
            corrupt.mkdir(parents=True, exist_ok=True)
            return str(corrupt)
        if val in ("<output_dir>", "<fast_out_dir>"):
            out = self.tmp_path / "out"
            out.mkdir(parents=True, exist_ok=True)
            return str(out)
        
        # Replace inline placeholders if any
        res = val
        res = res.replace("<host>", self.host)
        res = res.replace("<port>", str(self.port))
        if self.lib_dir:
            res = res.replace("<lib_dir>", str(self.lib_dir))
        if self.monolith_path:
            res = res.replace("<monolith_path>", str(self.monolith_path))
        return res

    def cleanup(self):
        """Terminate any background child processes."""
        for p in self.processes:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
        self.processes.clear()
        self.stages.clear()


@pytest.fixture
def bdd_context(tmp_path: Path) -> Generator[BddContext, None, None]:
    ctx = BddContext(tmp_path)
    yield ctx
    ctx.cleanup()


@pytest.fixture(scope="session")
def playwright_instance():
    """Session-scoped Playwright instance."""
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            yield p
    except ImportError:
        yield None


@pytest.fixture
def browser_context(playwright_instance):
    """Function-scoped headless browser."""
    if playwright_instance is None:
        pytest.skip("Playwright is not installed")
    try:
        browser = playwright_instance.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"]
        )
        page = browser.new_page()
        yield page
        browser.close()
    except Exception as exc:
        pytest.skip(f"Playwright browser could not be launched on this host ({exc}).")


# Import step definitions
BDD_DIR = Path(__file__).resolve().parent
if str(BDD_DIR) not in sys.path:
    sys.path.insert(0, str(BDD_DIR))

from steps.common_steps import *
from steps.layer_library_steps import *
from steps.serve_library_steps import *
from steps.stage_ring_steps import *
from steps.web_ui_steps import *
