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
GENIESIM_STRICT_RUNTIME_READINESS=${GENIESIM_STRICT_RUNTIME_READINESS:-0}
GENIESIM_STRICT_FAIL_ACTION=${GENIESIM_STRICT_FAIL_ACTION:-error}
GENIESIM_KEEP_OBJECTS_KINEMATIC=${GENIESIM_KEEP_OBJECTS_KINEMATIC:-0}
GENIESIM_SERVER_CUROBO_MODE=${GENIESIM_SERVER_CUROBO_MODE:-off}
GENIESIM_CUROBO_FAILOVER_FLAG=${GENIESIM_CUROBO_FAILOVER_FLAG:-${REPO_ROOT}/.geniesim_curobo_failover.flag}

_RUNTIME_READINESS_DEFAULT="/tmp/geniesim_runtime_readiness.json"
_RUNTIME_PRESTART_DEFAULT="/tmp/geniesim_runtime_prestart_probe.json"
if [ -d "/workspace/BlueprintPipeline" ]; then
  _RUNTIME_READINESS_DEFAULT="/workspace/BlueprintPipeline/.geniesim_runtime_readiness.json"
  _RUNTIME_PRESTART_DEFAULT="/workspace/BlueprintPipeline/.geniesim_runtime_prestart_probe.json"
fi
GENIESIM_RUNTIME_READINESS_JSON=${GENIESIM_RUNTIME_READINESS_JSON:-${_RUNTIME_READINESS_DEFAULT}}
GENIESIM_RUNTIME_PRESTART_JSON=${GENIESIM_RUNTIME_PRESTART_JSON:-${_RUNTIME_PRESTART_DEFAULT}}

if [ ! -x "${ISAAC_SIM_PATH}/python.sh" ]; then
  echo "[geniesim] ERROR: Isaac Sim not found at ${ISAAC_SIM_PATH}." >&2
  exit 1
fi

if [ ! -d "${GENIESIM_ROOT}/.git" ]; then
  echo "[geniesim] ERROR: Genie Sim not found at ${GENIESIM_ROOT}." >&2
  echo "          Use the pre-baked image (Dockerfile.geniesim-server) or run bootstrap_geniesim_runtime.sh instead." >&2
  exit 1
fi

# Remove stale raw-only flag files from previous sessions so that new runs
# start with a clean slate (STRICT_RUNTIME_PRECHECK_FAILED fix).
RAW_ONLY_FLAG_FILE="${GENIESIM_FORCE_RAW_ONLY_FLAG_FILE:-/tmp/geniesim_force_raw_only.flag}"
if [ -f "${RAW_ONLY_FLAG_FILE}" ]; then
  NOW_S="$(date +%s)"
  # Linux: stat -c %Y, BSD/macOS: stat -f %m
  MTIME_S="$(stat -c %Y "${RAW_ONLY_FLAG_FILE}" 2>/dev/null || stat -f %m "${RAW_ONLY_FLAG_FILE}" 2>/dev/null || true)"
  if [ -n "${MTIME_S}" ]; then
    AGE_S="$((NOW_S - MTIME_S))"
    STALE_THRESHOLD_S=600
    if [ "${AGE_S}" -gt "${STALE_THRESHOLD_S}" ]; then
      echo "[start_geniesim_server] Removing stale raw-only flag (${AGE_S}s old): ${RAW_ONLY_FLAG_FILE}"
      rm -f "${RAW_ONLY_FLAG_FILE}" || true
    else
      echo "[start_geniesim_server] Raw-only flag present (${AGE_S}s old); keeping: ${RAW_ONLY_FLAG_FILE}"
    fi
  else
    echo "[start_geniesim_server] Raw-only flag present but mtime unavailable; keeping: ${RAW_ONLY_FLAG_FILE}"
  fi
fi

export GENIESIM_ROOT
export ISAAC_SIM_PATH
export GENIESIM_RUNTIME_READINESS_JSON
export GENIESIM_RUNTIME_PRESTART_JSON
export GENIESIM_KEEP_OBJECTS_KINEMATIC
export GENIESIM_STRICT_RUNTIME_READINESS
export GENIESIM_STRICT_FAIL_ACTION
export GENIESIM_SERVER_CUROBO_MODE
export GENIESIM_CUROBO_FAILOVER_FLAG

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
_critical_patch_failures=()
_check_patch_marker() {
  local _file_path="$1"
  local _marker="$2"

  if ! grep -qF "${_marker}" "${_file_path}" 2>/dev/null; then
    _missing_patches+=("${_marker} (${_file_path})")
  fi
}

_check_patch_marker_any() {
  local _file_path="$1"
  shift
  local _marker
  for _marker in "$@"; do
    if grep -qF "${_marker}" "${_file_path}" 2>/dev/null; then
      return 0
    fi
  done
  _missing_patches+=("missing any marker [$*] (${_file_path})")
  return 1
}

_apply_patch_script() {
  local _patch_path="$1"
  local _label="$2"
  local _critical="$3"

  if [ ! -f "${_patch_path}" ]; then
    if [ "${_critical}" = "1" ]; then
      _critical_patch_failures+=("${_label}: missing script ${_patch_path}")
    else
      echo "[geniesim] WARNING: Optional patch script missing: ${_patch_path}" >&2
    fi
    return
  fi

  if ! "${ISAAC_SIM_PATH}/python.sh" "${_patch_path}" 2>&1; then
    if [ "${_critical}" = "1" ]; then
      _critical_patch_failures+=("${_label}: execution failed")
    else
      echo "[geniesim] WARNING: Optional patch failed: ${_label}" >&2
    fi
  fi
}

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

PATCHES_DIR="/workspace/BlueprintPipeline/tools/geniesim_adapter/deployment/patches"
if [ -d "${PATCHES_DIR}" ]; then
  echo "[geniesim] Applying runtime patches (GENIESIM_ROOT=${GENIESIM_ROOT})..."

  # Optional quality/compat patches
  _apply_patch_script "${PATCHES_DIR}/patch_camera_handler.py" "camera_handler" "0"
  _apply_patch_script "${PATCHES_DIR}/patch_observation_cameras.py" "observation_cameras" "0"
  _apply_patch_script "${PATCHES_DIR}/patch_object_pose_handler.py" "object_pose_handler" "0"
  _apply_patch_script "${PATCHES_DIR}/patch_ee_pose_handler.py" "ee_pose_handler" "0"
  _apply_patch_script "${PATCHES_DIR}/patch_data_collector_render_config.py" "render_config" "0"

  # Critical strict chain: real contacts, real efforts, real dynamics.
  _apply_patch_script "${PATCHES_DIR}/patch_contact_report.py" "contact_report" "1"
  _apply_patch_script "${PATCHES_DIR}/patch_joint_efforts_handler.py" "joint_efforts_handler" "1"
  _apply_patch_script "${PATCHES_DIR}/patch_enable_contacts_on_init.py" "enable_contacts_on_init" "1"
  _apply_patch_script "${PATCHES_DIR}/patch_sim_thread_physics_cache.py" "sim_thread_physics_cache" "1"
  _apply_patch_script "${PATCHES_DIR}/patch_register_scene_objects.py" "register_scene_objects" "1"
  _apply_patch_script "${PATCHES_DIR}/patch_deferred_dynamic_restore.py" "deferred_dynamic_restore" "1"
  _apply_patch_script "${PATCHES_DIR}/patch_dynamic_teleport_v5.py" "dynamic_teleport_v5" "1"
  _apply_patch_script "${PATCHES_DIR}/patch_fix_dynamic_prims_overwrite.py" "fix_dynamic_prims_overwrite" "1"
  _apply_patch_script "${PATCHES_DIR}/patch_scene_collision.py" "scene_collision" "1"
  _apply_patch_script "${PATCHES_DIR}/patch_ui_builder_time_import.py" "ui_builder_time_import" "1"
  _apply_patch_script "${PATCHES_DIR}/patch_curobo_config_type.py" "curobo_config_type_fix" "0"

  if [ "${GENIESIM_KEEP_OBJECTS_KINEMATIC}" = "1" ]; then
    echo "[geniesim] GENIESIM_KEEP_OBJECTS_KINEMATIC=1 — applying keep_kinematic patch"
    _apply_patch_script "${PATCHES_DIR}/patch_keep_objects_kinematic.py" "keep_objects_kinematic" "0"
  else
    echo "[geniesim] GENIESIM_KEEP_OBJECTS_KINEMATIC=0 — leaving dynamic object motion enabled"
  fi
  _apply_patch_script "${PATCHES_DIR}/patch_dynamic_grasp_toggle.py" "dynamic_grasp_toggle" "1"

  _startup_strict=0
  if [ "${GENIESIM_PATCH_CHECK_STRICT}" = "1" ] || [ "${GENIESIM_STRICT_RUNTIME_READINESS}" = "1" ]; then
    _startup_strict=1
  fi
  if [ "${#_critical_patch_failures[@]}" -gt 0 ]; then
    echo "[geniesim] ERROR: Critical runtime patch application failed:" >&2
    for _failure in "${_critical_patch_failures[@]}"; do
      echo "[geniesim] ERROR:   - ${_failure}" >&2
    done
    if [ "${_startup_strict}" = "1" ]; then
      exit 1
    fi
  fi

  _GRPC_SERVER="${GENIESIM_ROOT}/source/data_collection/server/grpc_server.py"
  _CMD_CTRL="${GENIESIM_ROOT}/source/data_collection/server/command_controller.py"
  _DCS_SERVER="${GENIESIM_ROOT}/source/data_collection/scripts/data_collector_server.py"

  _missing_patches=()
  _check_patch_marker "${_GRPC_SERVER}" "BlueprintPipeline contact_report patch"
  _check_patch_marker_any "${_GRPC_SERVER}" \
    "BlueprintPipeline joint_efforts patch" \
    "BlueprintPipeline joint_efforts handler patch"
  _check_patch_marker "${_CMD_CTRL}" "BlueprintPipeline contact_reporting_on_init patch"
  _check_patch_marker "${_CMD_CTRL}" "BlueprintPipeline sim_thread_physics_cache patch"
  _check_patch_marker "${_CMD_CTRL}" "BPv3_pre_play_kinematic"
  _check_patch_marker "${_CMD_CTRL}" "BPv4_deferred_dynamic_restore"
  _check_patch_marker "${_CMD_CTRL}" "BPv5_dynamic_teleport_usd_objects"
  _check_patch_marker "${_CMD_CTRL}" "BPv6_fix_dynamic_prims"
  _check_patch_marker "${_CMD_CTRL}" "BPv_dynamic_grasp_toggle"
  _check_patch_marker "${_GRPC_SERVER}" "BPv_dynamic_grasp_toggle"
  _check_patch_marker "${_CMD_CTRL}" "BlueprintPipeline object_pose patch"
  _check_patch_marker "${_CMD_CTRL}" "object_pose_resolver_v4"
  _check_patch_marker "${_CMD_CTRL}" "[PATCH] scene_collision_injected"
  _check_patch_marker "${_DCS_SERVER}" "BlueprintPipeline render config patch"
  if [ "${GENIESIM_KEEP_OBJECTS_KINEMATIC}" != "1" ] \
    && grep -qF "BPv7_keep_kinematic" "${_CMD_CTRL}" 2>/dev/null; then
    _missing_patches+=("forbidden marker BPv7_keep_kinematic present (${_CMD_CTRL})")
  fi
  _resolver_marker="$(grep -oE 'object_pose_resolver_v[0-9]+' "${_CMD_CTRL}" 2>/dev/null | head -1 || true)"
  if [ -n "${_resolver_marker}" ]; then
    echo "[geniesim] object_pose resolver marker: ${_resolver_marker}"
  fi
  if [ "${#_missing_patches[@]}" -gt 0 ]; then
    echo "[geniesim] ERROR: Runtime patch marker verification failed:" >&2
    for _missing in "${_missing_patches[@]}"; do
      echo "[geniesim] ERROR:   - ${_missing}" >&2
    done
    if [ "${_startup_strict}" = "1" ]; then
      exit 1
    fi
  fi

  if [ -f "${SCRIPT_DIR}/readiness_probe.py" ]; then
    _prestart_probe_args=(
      --skip-grpc
      --check-patches
      --geniesim-root "${GENIESIM_ROOT}"
      --require-patch-set strict
      --forbid-patch-set strict
      --output "${GENIESIM_RUNTIME_PRESTART_JSON}"
    )
    echo "[geniesim] Running pre-start patch readiness probe"
    if ! "${ISAAC_SIM_PATH}/python.sh" "${SCRIPT_DIR}/readiness_probe.py" "${_prestart_probe_args[@]}"; then
      echo "[geniesim] ERROR: Pre-start patch readiness probe failed." >&2
      if [ "${_startup_strict}" = "1" ]; then
        exit 1
      fi
    fi
  fi
fi

_SERVER_BASE_ARGS=""
[ "${GENIESIM_HEADLESS}" = "1" ] && _SERVER_BASE_ARGS="${_SERVER_BASE_ARGS} --headless"
# Always enable physics for dynamic rigid bodies
_SERVER_BASE_ARGS="${_SERVER_BASE_ARGS} --enable_physics"
# Only pass --publish_ros when ROS 2 is actually available
if [ "${GENIESIM_SKIP_ROS_RECORDING:-0}" != "1" ]; then
  _SERVER_BASE_ARGS="${_SERVER_BASE_ARGS} --publish_ros"
fi

_curobo_mode_raw="$(printf '%s' "${GENIESIM_SERVER_CUROBO_MODE}" | tr '[:upper:]' '[:lower:]')"
case "${_curobo_mode_raw}" in
  auto|on|off) ;;
  *)
    echo "[geniesim] WARNING: invalid GENIESIM_SERVER_CUROBO_MODE='${GENIESIM_SERVER_CUROBO_MODE}', defaulting to auto"
    _curobo_mode_raw="auto"
    ;;
esac
echo "[geniesim] cuRobo startup mode: ${_curobo_mode_raw} (flag: ${GENIESIM_CUROBO_FAILOVER_FLAG})"
_curobo_auto_failover_active=0

_resolve_server_args() {
  local _launch_mode="${_curobo_mode_raw}"
  local _args="${_SERVER_BASE_ARGS}"

  if [ "${_launch_mode}" = "auto" ]; then
    if [ "${_curobo_auto_failover_active}" = "1" ]; then
      _launch_mode="off"
    elif [ -f "${GENIESIM_CUROBO_FAILOVER_FLAG}" ]; then
      _launch_mode="off"
      _curobo_auto_failover_active=1
      rm -f "${GENIESIM_CUROBO_FAILOVER_FLAG}" 2>/dev/null || true
      echo "[geniesim] cuRobo auto-failover requested — launching without --enable_curobo"
    else
      _launch_mode="on"
    fi
  fi

  if [ "${_launch_mode}" = "on" ]; then
    _args="${_args} --enable_curobo"
  fi

  printf '%s' "${_args}"
}

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
  _SERVER_ARGS="$(_resolve_server_args)"
  echo "[geniesim] $(date '+%Y-%m-%d %H:%M:%S') Launching server (restart #${_restart_count}, logs: ${GENIESIM_SERVER_LOG})"
  echo "[geniesim] Launch args: ${_SERVER_ARGS}"

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
