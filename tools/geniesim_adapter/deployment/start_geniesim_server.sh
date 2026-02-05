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
GENIESIM_MAX_SERVER_RESTARTS=${GENIESIM_MAX_SERVER_RESTARTS:-3}
GENIESIM_PATCH_CHECK_STRICT=${GENIESIM_PATCH_CHECK_STRICT:-0}
GENIESIM_CAMERA_REQUIRE_DISPLAY=${GENIESIM_CAMERA_REQUIRE_DISPLAY:-1}

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

  if ! grep -qF "${_marker}" "${_file_path}" 2>/dev/null; then
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
if ! grep -qF "BlueprintPipeline joint_efforts patch" "${GENIESIM_ROOT}/source/data_collection/server/grpc_server.py" 2>/dev/null \
  && ! grep -qF "BlueprintPipeline joint_efforts handler patch" "${GENIESIM_ROOT}/source/data_collection/server/grpc_server.py" 2>/dev/null; then
  _missing_patches+=("BlueprintPipeline joint_efforts patch (${GENIESIM_ROOT}/source/data_collection/server/grpc_server.py)")
fi
_check_patch_marker \
  "${GENIESIM_ROOT}/source/data_collection/server/command_controller.py" \
  "BlueprintPipeline camera patch"
_check_patch_marker \
  "${GENIESIM_ROOT}/source/data_collection/server/command_controller.py" \
  "BlueprintPipeline object_pose patch"
_check_patch_marker \
  "${GENIESIM_ROOT}/source/data_collection/scripts/data_collector_server.py" \
  "BlueprintPipeline render config patch"

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

# ── Patch data_collector_server.py for headless RGB rendering ──
# The server ships with RealTimePathTracing + rt2 enabled, which doesn't produce
# RGB in headless mode.  We patch it at startup to:
#   1. Use headless=False when DISPLAY is set (so --no-window is NOT added)
#   2. Switch to RayTracedLighting renderer
#   3. Remove rt2/enabled=true from extra_args
#   4. Add --reset-user and --/renderer/activeGpu=0
_DCS="${GENIESIM_ROOT}/source/data_collection/scripts/data_collector_server.py"
if [ -f "${_DCS}" ] && ! grep -qF "os.environ.get(\"DISPLAY\")" "${_DCS}" 2>/dev/null; then
  echo "[geniesim] Patching data_collector_server.py for headless RGB rendering..."
  "${ISAAC_SIM_PATH}/python.sh" -c "
import os as _os, sys as _sys
path = '${_DCS}'
content = open(path).read()

# 1. Ensure 'import os' exists at the top (the original file uses os.path
#    without importing os — it relies on Isaac Sim launcher pre-importing it)
if 'import os' not in content.split('\\n')[0:10]:
    content = 'import os\\n' + content

# 2. headless=False when DISPLAY is set
content = content.replace(
    '\"headless\": args.headless',
    '\"headless\": False if os.environ.get(\"DISPLAY\") else args.headless')

# 3. Switch renderer
content = content.replace('\"RealTimePathTracing\"', '\"RayTracedLighting\"')

# 4. Remove rt2 flag from extra_args
import re
content = re.sub(r'\\s*\"--/persistent/rtx/modes/rt2/enabled=true\",?\\n?', '\\n', content)

# 5. Add --reset-user to extra args
if '\"--reset-user\"' not in content:
    content = content.replace(
        '\"extra_args\": [',
        '\"extra_args\": [\\n            \"--reset-user\",')

# 6. Disable rt2 via carb settings
if 'rt2/enabled\", False' not in content:
    content = content.replace(
        'simulation_app._carb_settings.set(\"/omni/replicator/asyncRendering\", False)',
        'simulation_app._carb_settings.set(\"/omni/replicator/asyncRendering\", False)\\nsimulation_app._carb_settings.set(\"/persistent/rtx/modes/rt2/enabled\", False)')

open(path, 'w').write(content)
print('[geniesim] data_collector_server.py patched for headless RGB')
" 2>&1
fi

# ── Apply camera handler patch ──
_CAMERA_PATCH="/workspace/BlueprintPipeline/tools/geniesim_adapter/deployment/patches/patch_camera_handler.py"
if [ -f "${_CAMERA_PATCH}" ]; then
  _CC="${GENIESIM_ROOT}/source/data_collection/server/command_controller.py"
  if ! grep -qF "BlueprintPipeline camera patch" "${_CC}" 2>/dev/null; then
    echo "[geniesim] Applying camera handler patch..."
    "${ISAAC_SIM_PATH}/python.sh" "${_CAMERA_PATCH}" 2>&1 || true
  fi
fi

_SERVER_ARGS=""
[ "${GENIESIM_HEADLESS}" = "1" ] && _SERVER_ARGS="${_SERVER_ARGS} --headless"
# Only pass --publish_ros when ROS 2 is actually available
if [ "${GENIESIM_SKIP_ROS_RECORDING:-0}" != "1" ]; then
  _SERVER_ARGS="${_SERVER_ARGS} --publish_ros"
fi

# ── Headless rendering setup ──
# Camera RGB rendering requires a display with NVIDIA GLX support.
# The preferred approach: docker-compose maps the host's Xorg X11 socket
# (/tmp/.X11-unix) into the container and sets DISPLAY=:99.  The host
# runs a headless Xorg on :99 using the nvidia driver.
#
# Fallback: if no DISPLAY is available, start Xvfb (software-only,
# no GLX — camera RGB will be black but depth/normals still work).
if [ "${GENIESIM_HEADLESS}" = "1" ]; then
  export ENABLE_CAMERAS="${ENABLE_CAMERAS:-1}"
  export SKIP_RGB_CAPTURE="${SKIP_RGB_CAPTURE:-}"

  _display_socket=""
  if [ -n "${DISPLAY:-}" ]; then
    _display_num="${DISPLAY##*:}"
    _display_num="${_display_num%%.*}"
    if [ -n "${_display_num}" ]; then
      _display_socket="/tmp/.X11-unix/X${_display_num}"
    fi
  fi

  if [ -n "${_display_socket}" ] && [ -S "${_display_socket}" ]; then
    echo "[geniesim] Using X display ${DISPLAY} (${_display_socket}) for camera rendering"
  else
    if [ "${ENABLE_CAMERAS}" = "1" ] && [ "${GENIESIM_CAMERA_REQUIRE_DISPLAY}" = "1" ]; then
      echo "[geniesim] ERROR: Camera rendering requires a valid X display socket." >&2
      echo "[geniesim] ERROR: DISPLAY='${DISPLAY:-<unset>}' socket='${_display_socket:-<none>}'" >&2
      echo "[geniesim] ERROR: Start host Xorg (:99) or set GENIESIM_CAMERA_REQUIRE_DISPLAY=0 for degraded Xvfb fallback." >&2
      exit 1
    fi

    if command -v Xvfb >/dev/null 2>&1; then
      echo "[geniesim] WARNING: No host X display found — using degraded Xvfb fallback (camera RGB may be black)"
      Xvfb :99 -screen 0 1280x720x24 +extension GLX +render -noreset &>/dev/null &
      export DISPLAY=:99
      sleep 1
    else
      echo "[geniesim] WARNING: No display available and Xvfb not installed — camera RGB may be black"
    fi
  fi
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
