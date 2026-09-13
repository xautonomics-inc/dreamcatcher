#!/usr/bin/env bash
# run-local.sh — Run Dreamcatcher BDD test suite locally or in GitLab docker runner
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ENV_FILE="${1:-${BDD_ENV_FILE:-}}"
if [[ -n "${ENV_FILE}" && -f "${ENV_FILE}" ]]; then
    echo "[bdd] Sourcing environment configuration from: ${ENV_FILE}"
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
fi

export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}:${REPO_ROOT}/tools/layer-distribution:${HOME}/.pylib"
export BDD_HOST="${BDD_HOST:-127.0.0.1}"
export BDD_PORT="${BDD_PORT:-8080}"

REPORT_XML="${SCRIPT_DIR}/report.xml"
TEST_TARGET="${BDD_TEST_TARGET:-${SCRIPT_DIR}}"

echo "============================================================"
echo " Running Dreamcatcher BDD Scenarios (pytest-bdd + Playwright)"
echo "============================================================"

# Capture pytest exit code without masking via || true
set +e
python3 -m pytest "${TEST_TARGET}" \
    -v \
    --junitxml="${REPORT_XML}" \
    -W ignore::pytest.PytestUnknownMarkWarning
PYTEST_EXIT=$?
set -e

if [[ -f "${REPORT_XML}" ]]; then
    echo "[bdd] Compiling test execution results into tests/bdd/RESULTS.md"
    python3 "${SCRIPT_DIR}/generate_results_md.py" "${REPORT_XML}"
fi

echo "[bdd] BDD run finished with exit code ${PYTEST_EXIT}."
exit "${PYTEST_EXIT}"
