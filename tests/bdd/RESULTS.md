# Dreamcatcher BDD Acceptance Test Results

**Run Date:** `2026-09-13 18:12:57 UTC`  
**Test Definition:** [`8ab4aba9d55b57c8de78702b52d6d67ad9b2e3a6`](https://github.com/xautonomics-inc/dreamcatcher/commit/8ab4aba9d55b57c8de78702b52d6d67ad9b2e3a6)<br>
**Binary Source Commit:** `9dd757919f0c30cf6945603dce799ad46da5cdfb`<br>
**Environment:** `GitLab Docker Runner (CPU, Playwright Chromium)`  
**Model:** `ggml-org/tiny-llamas/stories260K.gguf@def3e2dd70df35ecbf6403ea347de4c5977220c1`  
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
| `web-ui` | active model name is displayed in the web ui | **XFAIL** | Known issue tracked upstream in meta tracker |
| `web-ui` | submit chat prompt and receive streaming completion in browser | **PASSED** | - |
| `web-ui` | open settings dialog and verify configuration controls | **PASSED** | - |
