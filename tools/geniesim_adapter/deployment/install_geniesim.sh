#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

GENIESIM_ROOT=${GENIESIM_ROOT:-/opt/geniesim}
ISAAC_SIM_PATH=${ISAAC_SIM_PATH:-/isaac-sim}
GENIESIM_REPO=${GENIESIM_REPO:-https://github.com/AgibotTech/genie_sim.git}

# Determine the Genie Sim ref to checkout (env var > GENIESIM_REF file > none)
GENIESIM_REF=${GENIESIM_REF:-}
if [ -z "${GENIESIM_REF}" ]; then
  REF_FILE="${REPO_ROOT}/genie-sim-gpu-job/GENIESIM_REF"
  if [ -f "${REF_FILE}" ]; then
    GENIESIM_REF="$(tr -d '[:space:]' < "${REF_FILE}")"
    echo "[geniesim] Using GENIESIM_REF from ${REF_FILE}: ${GENIESIM_REF}"
  fi
fi

# Always ensure build tools are installed (needed for compiling Python packages)
if ! command -v git &>/dev/null || ! command -v g++ &>/dev/null || ! command -v cmake &>/dev/null; then
  echo "[geniesim] Installing build dependencies (git, g++, cmake)"
  apt-get update -qq && apt-get install -y -qq git build-essential cmake >/dev/null
fi

if [ ! -d "${GENIESIM_ROOT}/.git" ]; then
  echo "[geniesim] Cloning Genie Sim into ${GENIESIM_ROOT}"
  rm -rf "${GENIESIM_ROOT:?}"/* 2>/dev/null || true
  mkdir -p "${GENIESIM_ROOT}"
  git clone "${GENIESIM_REPO}" "${GENIESIM_ROOT}"
else
  echo "[geniesim] Genie Sim already present at ${GENIESIM_ROOT}"
fi

# Checkout specific ref if set
if [ -n "${GENIESIM_REF}" ]; then
  echo "[geniesim] Checking out GENIESIM_REF=${GENIESIM_REF}"
  cd "${GENIESIM_ROOT}"
  git fetch origin "${GENIESIM_REF}" 2>/dev/null || git fetch --unshallow 2>/dev/null || true
  git checkout "${GENIESIM_REF}"
  cd "${OLDPWD}"
fi

if [ ! -x "${ISAAC_SIM_PATH}/python.sh" ]; then
  echo "[geniesim] ERROR: Isaac Sim not found at ${ISAAC_SIM_PATH}."
  echo "          Set ISAAC_SIM_PATH to your Isaac Sim install before continuing."
  exit 1
fi

if [ -f "${GENIESIM_ROOT}/requirements.txt" ]; then
  echo "[geniesim] Installing Genie Sim Python dependencies via Isaac Sim Python"
  "${ISAAC_SIM_PATH}/python.sh" -m pip install --quiet --upgrade pip 2>&1 | tail -3
  # Allow partial failures — some Genie Sim deps may need optional system libs
  "${ISAAC_SIM_PATH}/python.sh" -m pip install -r "${GENIESIM_ROOT}/requirements.txt" || \
    echo "[geniesim] WARNING: Some Genie Sim dependencies failed to install (non-fatal)" >&2
else
  echo "[geniesim] WARNING: requirements.txt not found under ${GENIESIM_ROOT}" >&2
fi

cat <<SUMMARY
[geniesim] Installation complete.
- GENIESIM_ROOT=${GENIESIM_ROOT}
- ISAAC_SIM_PATH=${ISAAC_SIM_PATH}

Next steps:
  export GENIESIM_ROOT=${GENIESIM_ROOT}
  export ISAAC_SIM_PATH=${ISAAC_SIM_PATH}
  export GENIESIM_HOST=localhost
  export GENIESIM_PORT=50051

Then start the server:
  ${ISAAC_SIM_PATH}/python.sh ${GENIESIM_ROOT}/source/data_collection/scripts/data_collector_server.py --headless --port ${GENIESIM_PORT}
SUMMARY
