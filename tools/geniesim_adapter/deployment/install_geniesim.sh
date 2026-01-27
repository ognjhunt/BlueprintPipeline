#!/usr/bin/env bash
set -euo pipefail

GENIESIM_ROOT=${GENIESIM_ROOT:-/opt/geniesim}
ISAAC_SIM_PATH=${ISAAC_SIM_PATH:-/isaac-sim}
GENIESIM_REPO=${GENIESIM_REPO:-https://github.com/AgibotTech/genie_sim.git}

if [ ! -d "${GENIESIM_ROOT}/.git" ]; then
  if ! command -v git &>/dev/null; then
    echo "[geniesim] Installing git"
    apt-get update -qq && apt-get install -y -qq git >/dev/null
  fi
  echo "[geniesim] Cloning Genie Sim into ${GENIESIM_ROOT}"
  rm -rf "${GENIESIM_ROOT:?}"/* 2>/dev/null || true
  mkdir -p "${GENIESIM_ROOT}"
  git clone "${GENIESIM_REPO}" "${GENIESIM_ROOT}"
else
  echo "[geniesim] Genie Sim already present at ${GENIESIM_ROOT}"
fi

if [ ! -x "${ISAAC_SIM_PATH}/python.sh" ]; then
  echo "[geniesim] ERROR: Isaac Sim not found at ${ISAAC_SIM_PATH}."
  echo "          Set ISAAC_SIM_PATH to your Isaac Sim install before continuing."
  exit 1
fi

if [ -f "${GENIESIM_ROOT}/requirements.txt" ]; then
  echo "[geniesim] Installing Genie Sim Python dependencies via Isaac Sim Python"
  "${ISAAC_SIM_PATH}/python.sh" -m pip install --upgrade pip
  "${ISAAC_SIM_PATH}/python.sh" -m pip install -r "${GENIESIM_ROOT}/requirements.txt"
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
