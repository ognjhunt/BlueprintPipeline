#!/usr/bin/env bash
set -euo pipefail

# Lightweight entrypoint for the pre-baked geniesim-server image.
# All dependencies are already installed — this script only starts the server.
# Includes a restart loop: if the server crashes, it auto-restarts up to
# GENIESIM_MAX_SERVER_RESTARTS times (default 1).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

GENIESIM_ROOT=${GENIESIM_ROOT:-/opt/geniesim}
ISAAC_SIM_PATH=${ISAAC_SIM_PATH:-/isaac-sim}
GENIESIM_PORT=${GENIESIM_PORT:-50051}
GENIESIM_HEADLESS=${GENIESIM_HEADLESS:-1}
GENIESIM_SERVER_LOG=${GENIESIM_SERVER_LOG:-/tmp/geniesim_server.log}
GENIESIM_MAX_SERVER_RESTARTS=${GENIESIM_MAX_SERVER_RESTARTS:-1}
GENIESIM_PATCH_CHECK_STRICT=${GENIESIM_PATCH_CHECK_STRICT:-0}

if [ ! -x "${ISAAC_SIM_PATH}/python.sh" ]; then
  echo "[geniesim] ERROR: Isaac Sim not found at ${ISAAC_SIM_PATH}." >&2
  exit 1
fi

if [ ! -d "${GENIESIM_ROOT}/.git" ]; then
  echo "[geniesim] ERROR: Genie Sim not found at ${GENIESIM_ROOT}." >&2
  echo "          Use the pre-baked image (Dockerfile.geniesim-server) or run bootstrap_geniesim_runtime.sh instead." >&2
  exit 1
fi

export GENIESIM_ROOT
export ISAAC_SIM_PATH

# Source pre-baked ROS 2 env if available, otherwise detect at runtime.
if [ -f /etc/geniesim-ros2.env ]; then
  # shellcheck disable=SC1091
  source /etc/geniesim-ros2.env
else
  _ROS2_DISTRO="humble"
  if [ -d "${ISAAC_SIM_PATH}/exts/isaacsim.ros2.bridge/jazzy/lib" ]; then
    _ROS2_DISTRO="jazzy"
  fi
  _ROS2_BASE="${ISAAC_SIM_PATH}/exts/isaacsim.ros2.bridge/${_ROS2_DISTRO}"
  export LD_LIBRARY_PATH="${_ROS2_BASE}/lib:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="${_ROS2_BASE}/rclpy:${_ROS2_BASE}:${PYTHONPATH:-}"
fi

export PYTHONPATH="${REPO_ROOT}/tools/geniesim_adapter:${REPO_ROOT}:${PYTHONPATH:-}"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

_missing_patches=()
_check_patch_marker() {
  local _file_path="$1"
  local _marker="$2"

  if ! rg --quiet --fixed-strings "${_marker}" "${_file_path}"; then
    _missing_patches+=("${_marker} (${_file_path})")
  fi
}

_check_patch_marker \
  "${GENIESIM_ROOT}/source/data_collection/server/grpc_server.py" \
  "BlueprintPipeline grpc_camera_info patch"
_check_patch_marker \
  "${GENIESIM_ROOT}/source/data_collection/server/command_controller.py" \
  "BlueprintPipeline observation_cameras patch"
_check_patch_marker \
  "${GENIESIM_ROOT}/source/data_collection/server/grpc_server.py" \
  "BlueprintPipeline contact_report patch"
if ! rg --quiet --fixed-strings "BlueprintPipeline joint_efforts patch" "${GENIESIM_ROOT}/source/data_collection/server/grpc_server.py" \
  && ! rg --quiet --fixed-strings "BlueprintPipeline joint_efforts handler patch" "${GENIESIM_ROOT}/source/data_collection/server/grpc_server.py"; then
  _missing_patches+=("BlueprintPipeline joint_efforts patch (${GENIESIM_ROOT}/source/data_collection/server/grpc_server.py)")
fi
_check_patch_marker \
  "${GENIESIM_ROOT}/source/data_collection/server/command_controller.py" \
  "BlueprintPipeline camera patch"
_check_patch_marker \
  "${GENIESIM_ROOT}/source/data_collection/server/command_controller.py" \
  "BlueprintPipeline object_pose patch"

if [ "${#_missing_patches[@]}" -gt 0 ]; then
  echo "[geniesim] WARNING: Missing expected BlueprintPipeline patch markers:" >&2
  for _missing in "${_missing_patches[@]}"; do
    echo "[geniesim] WARNING:   - ${_missing}" >&2
  done
  if [ "${GENIESIM_PATCH_CHECK_STRICT}" = "1" ]; then
    echo "[geniesim] ERROR: Patch marker verification failed; refusing to start." >&2
    exit 1
  fi
fi

_SERVER_ARGS=""
[ "${GENIESIM_HEADLESS}" = "1" ] && _SERVER_ARGS="${_SERVER_ARGS} --headless"
# Only pass --publish_ros when ROS 2 is actually available
if [ "${GENIESIM_SKIP_ROS_RECORDING:-0}" != "1" ]; then
  _SERVER_ARGS="${_SERVER_ARGS} --publish_ros"
fi

_restart_count=0

echo "[geniesim] Starting Genie Sim server with restart loop (max ${GENIESIM_MAX_SERVER_RESTARTS} restarts)"

while true; do
  echo "[geniesim] $(date '+%Y-%m-%d %H:%M:%S') Launching server (restart #${_restart_count}, logs: ${GENIESIM_SERVER_LOG})"

  "${ISAAC_SIM_PATH}/python.sh" \
    "${GENIESIM_ROOT}/source/data_collection/scripts/data_collector_server.py" \
    ${_SERVER_ARGS} \
    >> "${GENIESIM_SERVER_LOG}" 2>&1 &
  _PID=$!

  # Wait for server process to exit
  _exit_code=0
  wait $_PID || _exit_code=$?

  echo "[geniesim] $(date '+%Y-%m-%d %H:%M:%S') Server exited with code ${_exit_code}"

  _restart_count=$((_restart_count + 1))
  if [ "${_restart_count}" -gt "${GENIESIM_MAX_SERVER_RESTARTS}" ]; then
    echo "[geniesim] Max restarts (${GENIESIM_MAX_SERVER_RESTARTS}) exceeded — giving up"
    exit 1
  fi

  echo "[geniesim] Restarting server in 2s..."
  sleep 2
done
