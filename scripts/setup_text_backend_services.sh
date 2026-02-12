#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV_PATH=${TEXT_BACKEND_VENV_PATH:-"${REPO_ROOT}/.venv-text-backends"}
PYTHON_BIN=${PYTHON_BIN:-python3}

echo "Using python: ${PYTHON_BIN}"
echo "Creating venv: ${VENV_PATH}"
"${PYTHON_BIN}" -m venv "${VENV_PATH}"

"${VENV_PATH}/bin/pip" install --upgrade pip
"${VENV_PATH}/bin/pip" install -r "${REPO_ROOT}/scenesmith-service/requirements.txt"
"${VENV_PATH}/bin/pip" install -r "${REPO_ROOT}/sage-service/requirements.txt"

echo ""
echo "Setup complete."
echo "Use this python when starting services:"
echo "  export PYTHON_BIN=${VENV_PATH}/bin/python"
echo "  ./scripts/start_text_backend_services.sh start"
