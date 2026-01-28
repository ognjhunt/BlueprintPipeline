#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

GENIESIM_ROOT=${GENIESIM_ROOT:-/opt/geniesim}
ISAAC_SIM_PATH=${ISAAC_SIM_PATH:-/isaac-sim}
GENIESIM_HOST=${GENIESIM_HOST:-localhost}
GENIESIM_PORT=${GENIESIM_PORT:-50051}
GENIESIM_HEADLESS=${GENIESIM_HEADLESS:-1}
GENIESIM_START_SERVER=${GENIESIM_START_SERVER:-1}
GENIESIM_HEALTHCHECK=${GENIESIM_HEALTHCHECK:-1}
GENIESIM_SERVER_LOG=${GENIESIM_SERVER_LOG:-/tmp/geniesim_server.log}

if [ ! -x "${ISAAC_SIM_PATH}/python.sh" ]; then
  echo "[geniesim] ERROR: Isaac Sim not found at ${ISAAC_SIM_PATH}." >&2
  exit 1
fi

"${SCRIPT_DIR}/install_geniesim.sh"

# Install pipeline Python dependencies into Isaac Sim's Python
PIPELINE_REQS="${REPO_ROOT}/genie-sim-local-job/requirements.txt"
if [ -f "${PIPELINE_REQS}" ]; then
  echo "[geniesim] Installing pipeline Python dependencies"
  "${ISAAC_SIM_PATH}/python.sh" -m pip install --quiet -r "${PIPELINE_REQS}" 2>&1 | tail -10 || {
    echo "[geniesim] WARNING: Some pipeline dependencies failed (non-fatal)"
  }
fi

# Ensure cuRobo is available for Genie Sim motion planning.
# The nvidia-curobo PyPI package is a placeholder; cuRobo must be built from source.
echo "[geniesim] Ensuring cuRobo is installed"
if "${ISAAC_SIM_PATH}/python.sh" - <<'PY' >/dev/null 2>&1; then
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("curobo") else 1)
PY
  echo "[geniesim] cuRobo already installed"
else
  echo "[geniesim] Installing cuRobo from source (this may take ~20 minutes)..."
  CUROBO_DIR="${GENIESIM_ROOT}/curobo"
  if [ ! -d "${CUROBO_DIR}/.git" ]; then
    git clone https://github.com/NVlabs/curobo.git "${CUROBO_DIR}"
  fi
  cd "${CUROBO_DIR}"
  "${ISAAC_SIM_PATH}/python.sh" -m pip install -e . --no-build-isolation 2>&1 | tail -20 || {
    echo "[geniesim] WARNING: cuRobo source install failed (required for full motion planning)"
  }
  cd "${OLDPWD}"
fi

export GENIESIM_ROOT
export ISAAC_SIM_PATH
export GENIESIM_HOST
export GENIESIM_PORT
export PYTHONPATH="${REPO_ROOT}/tools/geniesim_adapter:${REPO_ROOT}:${PYTHONPATH:-}"

# Enable Isaac Sim's internal ROS 2 libraries for rclpy
# Genie Sim's command_controller.py imports rclpy unconditionally
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH="${ISAAC_SIM_PATH}/exts/isaacsim.ros2.bridge/humble/lib:${LD_LIBRARY_PATH:-}"

if [ "${GENIESIM_START_SERVER}" = "1" ]; then
  echo "[geniesim] Starting Genie Sim server (logs: ${GENIESIM_SERVER_LOG})"
  # Note: data_collector_server.py doesn't accept --host/--port args
  # It listens on its default port (50051)
  nohup "${ISAAC_SIM_PATH}/python.sh" \
    "${GENIESIM_ROOT}/source/data_collection/scripts/data_collector_server.py" \
    $( [ "${GENIESIM_HEADLESS}" = "1" ] && echo "--headless" ) \
    > "${GENIESIM_SERVER_LOG}" 2>&1 &
fi

if [ "${GENIESIM_HEALTHCHECK}" = "1" ]; then
  echo "[geniesim] Running health check (informational only)"
  "${ISAAC_SIM_PATH}/python.sh" -m tools.geniesim_adapter.geniesim_healthcheck || {
    echo "[geniesim] Health check not yet passing - server may still be starting"
  }
fi

# Keep container alive by waiting on the server process
if [ "${GENIESIM_START_SERVER}" = "1" ]; then
  echo "[geniesim] Waiting for server process..."
  # Wait for the server log to be created and then tail it
  while [ ! -f "${GENIESIM_SERVER_LOG}" ]; do sleep 1; done
  tail -f "${GENIESIM_SERVER_LOG}"
fi
