# Dreamcatcher BDD Acceptance Test Results

**Run Date:** `2026-09-13 18:26:04 UTC`<br>
**Test Definition:** [`60904a0c575961f5f5327605565876951cfc664b`](https://github.com/xautonomics-inc/dreamcatcher/commit/60904a0c575961f5f5327605565876951cfc664b)<br>
**Binary Source Commit:** `9c6c529121c54d067537bdd39758f497aba9afb3`<br>
**Environment:** `GitLab Docker Runner (CPU, Playwright Chromium)`<br>
**Model:** `ggml-org/tiny-llamas/stories260K.gguf@def3e2dd70df35ecbf6403ea347de4c5977220c1`<br>
**Status:** **GREEN (PASS + KNOWN ISSUES)**

---

## Summary

| Total Scenarios | Passed (Live) | Known Issues (XFail) | Skipped (Prerequisites) | Failed |
| :---: | :---: | :---: | :---: | :---: |
| **4** | **3** | **1** | **0** | **0** |

---

## Detailed Scenario Outcomes

| Feature | Scenario | Status | Reason / Notes |
| :--- | :--- | :---: | :--- |
| `web-ui` | web ui loads successfully in headless browser | **PASSED** | - |
| `web-ui` | active model name is displayed in the web ui | **XFAIL** | meta#100: --model-dir server reports empty model_name |
| `web-ui` | submit chat prompt and receive streaming completion in browser | **PASSED** | - |
| `web-ui` | open settings dialog and verify configuration controls | **PASSED** | - |
