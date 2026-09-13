#!/usr/bin/env python3
"""Parse pytest XML report and generate tests/bdd/RESULTS.md."""

from __future__ import annotations

import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_FILE = REPO_ROOT / "tests" / "bdd" / "RESULTS.md"


def get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def generate_results_markdown(xml_path: Path) -> str:
    if not xml_path.exists():
        sys.exit(f"Error: JUnit XML report not found at {xml_path}. Run tests/bdd/run-local.sh first.")

    binary_commit = os.environ.get("BDD_BINARY_COMMIT", get_git_commit())
    github_commit = os.environ.get("BDD_GITHUB_COMMIT", binary_commit)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    passed = 0
    skipped = 0
    xfailed = 0
    failed = 0
    test_cases: list[dict[str, str]] = []

    for tc in root.iter("testcase"):
        name = tc.get("name", "")
        classname = tc.get("classname", "")
        status = "PASSED"
        note = ""

        if tc.find("failure") is not None:
            status = "FAILED"
            failed += 1
            fail_elem = tc.find("failure")
            note = fail_elem.get("message", "") if fail_elem is not None else ""
        elif tc.find("skipped") is not None:
            sk = tc.find("skipped")
            msg = sk.get("message", "") if sk is not None else ""
            if "xfail" in msg.lower() or "meta#" in msg.lower() or "known issue" in msg.lower():
                status = "XFAIL"
                xfailed += 1
                note = msg
            else:
                status = "SKIPPED"
                skipped += 1
                note = msg
        else:
            passed += 1

        test_cases.append({
            "feature": classname.split(".")[-1],
            "scenario": name,
            "status": status,
            "note": note,
        })

    total = passed + xfailed + skipped + failed

    # Determine execution mode and overall status
    if failed > 0:
        overall_status = "RED (FAILED)"
    elif skipped > 0 and passed > 0:
        overall_status = "PARTIAL / PREREQUISITES_PENDING"
    elif skipped == total:
        overall_status = "SKIPPED (NO PREREQUISITES)"
    elif xfailed > 0:
        overall_status = "GREEN (PASS + KNOWN ISSUES)"
    else:
        overall_status = "GREEN (ALL PASSED)"

    model_used = os.environ.get("BDD_MODEL", "unbound")
    runner_env = os.environ.get("BDD_RUNNER", "Local Fleet Runner")

    md = f"""# Dreamcatcher BDD Acceptance Test Results

**Run Date:** `{timestamp}`<br>
**Test Definition:** [`{github_commit}`](https://github.com/xautonomics-inc/dreamcatcher/commit/{github_commit})<br>
**Binary Source Commit:** `{binary_commit}`<br>
**Environment:** `{runner_env}`<br>
**Model:** `{model_used}`<br>
**Status:** **{overall_status}**

---

## Summary

| Total Scenarios | Passed (Live) | Known Issues (XFail) | Skipped (Prerequisites) | Failed |
| :---: | :---: | :---: | :---: | :---: |
| **{total}** | **{passed}** | **{xfailed}** | **{skipped}** | **{failed}** |

---

## Detailed Scenario Outcomes

| Feature | Scenario | Status | Reason / Notes |
| :--- | :--- | :---: | :--- |
"""
    for tc in test_cases:
        feat = tc["feature"].replace("test_", "").replace("_", "-")
        scen = tc["scenario"].replace("test_", "").replace("_", " ")
        note = tc["note"].replace("\n", " ").strip() if tc["note"] else "-"
        md += f"| `{feat}` | {scen} | **{tc['status']}** | {note} |\n"

    return md


def main() -> int:
    if len(sys.argv) < 2:
        sys.exit("Usage: generate_results_md.py <path-to-report.xml>")
    xml_file = Path(sys.argv[1])
    content = generate_results_markdown(xml_file)
    RESULTS_FILE.write_text(content, encoding="utf-8")
    print(f"Wrote {RESULTS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
