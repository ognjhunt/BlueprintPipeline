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
GENIESIM_STRICT_RUNTIME_READINESS=${GENIESIM_STRICT_RUNTIME_READINESS:-0}
GENIESIM_STRICT_FAIL_ACTION=${GENIESIM_STRICT_FAIL_ACTION:-raw_only}
GENIESIM_RUNTIME_READINESS_JSON=${GENIESIM_RUNTIME_READINESS_JSON:-/tmp/geniesim_runtime_readiness.json}
GENIESIM_RUNTIME_PRESTART_JSON=${GENIESIM_RUNTIME_PRESTART_JSON:-/tmp/geniesim_runtime_prestart_probe.json}
GENIESIM_FORCE_RAW_ONLY_FLAG_FILE=${GENIESIM_FORCE_RAW_ONLY_FLAG_FILE:-/tmp/geniesim_force_raw_only.flag}

if [ ! -x "${ISAAC_SIM_PATH}/python.sh" ]; then
  echo "[geniesim] ERROR: Isaac Sim not found at ${ISAAC_SIM_PATH}." >&2
  exit 1
fi

_mark_force_raw_only() {
  local reason="${1:-unknown}"
  echo "[geniesim] Forcing raw-only certification mode: ${reason}"
  export GENIESIM_FORCE_RAW_ONLY=1
  export PHYSICS_CERT_MODE=compat
  echo "reason=${reason}" > "${GENIESIM_FORCE_RAW_ONLY_FLAG_FILE}"
}

# Fast-path: if running in a pre-baked image, skip installation entirely
if [ -f /opt/.geniesim-prebaked ]; then
  echo "[geniesim] Pre-baked image detected — skipping installation"
  exec "${SCRIPT_DIR}/start_geniesim_server.sh"
fi

"${SCRIPT_DIR}/install_geniesim.sh"

# ── Validate GENIESIM_ROOT and discover server files ──────────────────────────
# Patches target specific Python files inside the GenieSim source tree.
# If the expected paths don't exist (different repo layout, pre-installed at
# alternate location), discover them dynamically and update GENIESIM_ROOT.
_GRPC_EXPECTED="${GENIESIM_ROOT}/source/data_collection/server/grpc_server.py"
_CC_EXPECTED="${GENIESIM_ROOT}/source/data_collection/server/command_controller.py"
_DCS_EXPECTED="${GENIESIM_ROOT}/source/data_collection/scripts/data_collector_server.py"

_discover_geniesim_root() {
  # Strategy: find grpc_server.py under common locations and derive GENIESIM_ROOT
  local _search_dirs=("${GENIESIM_ROOT}" "/opt/geniesim" "/opt/nvidia/geniesim" "/workspace/geniesim" "/home")
  for _dir in "${_search_dirs[@]}"; do
    [ -d "${_dir}" ] || continue
    local _found
    _found="$(find "${_dir}" -maxdepth 6 -name "grpc_server.py" -path "*/data_collection/server/*" -print -quit 2>/dev/null)" || true
    if [ -n "${_found}" ]; then
      # Derive GENIESIM_ROOT: strip /source/data_collection/server/grpc_server.py
      local _server_dir
      _server_dir="$(dirname "${_found}")"              # .../server
      local _dc_dir
      _dc_dir="$(dirname "${_server_dir}")"             # .../data_collection
      local _source_dir
      _source_dir="$(dirname "${_dc_dir}")"             # .../source
      local _candidate
      _candidate="$(dirname "${_source_dir}")"          # GENIESIM_ROOT
      if [ -f "${_candidate}/source/data_collection/server/grpc_server.py" ]; then
        echo "${_candidate}"
        return 0
      fi
    fi
  done
  return 1
}

if [ ! -f "${_GRPC_EXPECTED}" ] || [ ! -f "${_CC_EXPECTED}" ]; then
  echo "[geniesim] WARNING: Expected server files not found at ${GENIESIM_ROOT}"
  echo "[geniesim] Searching for GenieSim source tree..."
  _DISCOVERED_ROOT="$(_discover_geniesim_root)" || _DISCOVERED_ROOT=""
  if [ -n "${_DISCOVERED_ROOT}" ] && [ "${_DISCOVERED_ROOT}" != "${GENIESIM_ROOT}" ]; then
    echo "[geniesim] ✓ Discovered GenieSim at ${_DISCOVERED_ROOT} (was: ${GENIESIM_ROOT})"
    GENIESIM_ROOT="${_DISCOVERED_ROOT}"
    export GENIESIM_ROOT
    _GRPC_EXPECTED="${GENIESIM_ROOT}/source/data_collection/server/grpc_server.py"
    _CC_EXPECTED="${GENIESIM_ROOT}/source/data_collection/server/command_controller.py"
    _DCS_EXPECTED="${GENIESIM_ROOT}/source/data_collection/scripts/data_collector_server.py"
  else
    echo "[geniesim] ERROR: Could not find GenieSim server files anywhere."
    echo "[geniesim] Looked for: grpc_server.py under data_collection/server/"
    echo "[geniesim] Patches will NOT be applied. Continuing with degraded mode."
    _mark_force_raw_only "geniesim_server_files_not_found"
  fi
fi

# Final pre-flight check: log which target files exist
echo "[geniesim] Pre-flight patch target validation:"
for _target_file in "${_GRPC_EXPECTED}" "${_CC_EXPECTED}" "${_DCS_EXPECTED}"; do
  if [ -f "${_target_file}" ]; then
    echo "[geniesim]   ✓ ${_target_file}"
  else
    echo "[geniesim]   ✗ ${_target_file} (MISSING)"
  fi
done

# Apply server patches (render config, camera handler, object pose, ee pose, omnigraph dedup)
PATCHES_DIR="${SCRIPT_DIR}/patches"
if [ -d "${PATCHES_DIR}" ]; then
  echo "[geniesim] Applying server patches (GENIESIM_ROOT=${GENIESIM_ROOT})..."
  for patch_script in \
    "${PATCHES_DIR}/patch_omnigraph_dedup.py" \
    "${PATCHES_DIR}/patch_data_collector_render_config.py" \
    "${PATCHES_DIR}/patch_camera_handler.py" \
    "${PATCHES_DIR}/patch_object_pose_handler.py" \
    "${PATCHES_DIR}/patch_ee_pose_handler.py" \
    "${PATCHES_DIR}/patch_stage_diagnostics.py" \
    "${PATCHES_DIR}/patch_observation_cameras.py" \
    "${PATCHES_DIR}/patch_contact_report.py" \
    "${PATCHES_DIR}/patch_joint_efforts_handler.py" \
    "${PATCHES_DIR}/patch_ee_wrench_handler.py" \
    "${PATCHES_DIR}/patch_grpc_server.py" \
    "${PATCHES_DIR}/_apply_safe_float.py" \
    "${PATCHES_DIR}/patch_xforms_safe_rotation.py" \
    "${PATCHES_DIR}/patch_articulation_guard.py" \
    "${PATCHES_DIR}/patch_enable_contacts_on_init.py" \
    "${PATCHES_DIR}/patch_articulation_physics_wait.py" \
    "${PATCHES_DIR}/patch_set_joint_guard.py" \
    "${PATCHES_DIR}/patch_camera_crash_guard.py" \
    "${PATCHES_DIR}/patch_sim_thread_physics_cache.py" \
    "${PATCHES_DIR}/patch_autoplay.py" \
    "${PATCHES_DIR}/patch_register_scene_objects.py" \
    "${PATCHES_DIR}/patch_deferred_dynamic_restore.py" \
    "${PATCHES_DIR}/patch_dynamic_teleport_v5.py" \
    "${PATCHES_DIR}/patch_fix_dynamic_prims_overwrite.py"; do
    if [ -f "${patch_script}" ]; then
      "${ISAAC_SIM_PATH}/python.sh" "${patch_script}" || echo "[geniesim] WARNING: ${patch_script} failed (non-fatal)"
    fi
  done

  # Verify critical patches were applied
  echo "[geniesim] Verifying patch application..."
  _GRPC_SERVER="${GENIESIM_ROOT}/source/data_collection/server/grpc_server.py"
  _CMD_CTRL="${GENIESIM_ROOT}/source/data_collection/server/command_controller.py"
  if grep -q "BlueprintPipeline contact_report patch" "${_GRPC_SERVER}" 2>/dev/null; then
    echo "[geniesim] ✓ contact_report patch verified in grpc_server.py"
  else
    echo "[geniesim] WARNING: contact_report patch may not have been applied to grpc_server.py"
  fi
  if grep -q "BlueprintPipeline object_pose patch" "${_CMD_CTRL}" 2>/dev/null; then
    echo "[geniesim] ✓ object_pose patch verified in command_controller.py"
  else
    echo "[geniesim] WARNING: object_pose patch may not have been applied to command_controller.py"
  fi
  if grep -q "BlueprintPipeline sim_thread_physics_cache patch" "${_CMD_CTRL}" 2>/dev/null; then
    echo "[geniesim] ✓ sim_thread_physics_cache patch verified in command_controller.py"
  else
    echo "[geniesim] WARNING: sim_thread_physics_cache patch may not have been applied"
  fi

  # Pre-start patch marker readiness probe
  if [ -f "${SCRIPT_DIR}/readiness_probe.py" ]; then
    echo "[geniesim] Running pre-start runtime patch readiness probe"
    if ! "${ISAAC_SIM_PATH}/python.sh" "${SCRIPT_DIR}/readiness_probe.py" \
      --skip-grpc \
      --check-patches \
      --geniesim-root "${GENIESIM_ROOT}" \
      --output "${GENIESIM_RUNTIME_PRESTART_JSON}"; then
      echo "[geniesim] WARNING: Runtime patch pre-start readiness probe failed"
      if [ "${GENIESIM_STRICT_RUNTIME_READINESS}" = "1" ]; then
        if [ "${GENIESIM_STRICT_FAIL_ACTION}" = "error" ]; then
          echo "[geniesim] ERROR: strict runtime readiness requested and pre-start patch probe failed"
          exit 1
        fi
        _mark_force_raw_only "runtime_patch_prestart_probe_failed"
      fi
    fi
  fi
fi

# Sync updated gRPC proto/stubs into the runtime (ensures new RPCs are registered)
_GRPC_SRC_DIR="${REPO_ROOT}/tools/geniesim_adapter"
_GRPC_PB2="${_GRPC_SRC_DIR}/geniesim_grpc_pb2.py"
_GRPC_PB2_GRPC="${_GRPC_SRC_DIR}/geniesim_grpc_pb2_grpc.py"
_GRPC_PROTO="${_GRPC_SRC_DIR}/geniesim_grpc.proto"
if [ -f "${_GRPC_PB2}" ] && [ -f "${_GRPC_PB2_GRPC}" ]; then
  echo "[geniesim] Syncing gRPC stubs/proto into GENIESIM_ROOT..."
  _grpc_dirs="$(find "${GENIESIM_ROOT}" -type f \
      \( -name "geniesim_grpc_pb2.py" -o -name "geniesim_grpc_pb2_grpc.py" -o -name "geniesim_grpc.proto" \) \
      -print 2>/dev/null | xargs -r -n1 dirname | sort -u)"
  _synced=0
  if [ -n "${_grpc_dirs}" ]; then
    while IFS= read -r _dir; do
      [ -z "${_dir}" ] && continue
      cp -f "${_GRPC_PB2}" "${_dir}/" || true
      cp -f "${_GRPC_PB2_GRPC}" "${_dir}/" || true
      [ -f "${_GRPC_PROTO}" ] && cp -f "${_GRPC_PROTO}" "${_dir}/" || true
      _synced=$((_synced + 1))
    done <<< "${_grpc_dirs}"
  fi
  _fallback_dir="${GENIESIM_ROOT}/source/data_collection/aimdk/protocol"
  if [ "${_synced}" -eq 0 ] && [ -d "${_fallback_dir}" ]; then
    cp -f "${_GRPC_PB2}" "${_fallback_dir}/" || true
    cp -f "${_GRPC_PB2_GRPC}" "${_fallback_dir}/" || true
    [ -f "${_GRPC_PROTO}" ] && cp -f "${_GRPC_PROTO}" "${_fallback_dir}/" || true
    _synced=1
  fi
  if [ "${_synced}" -gt 0 ]; then
    echo "[geniesim] ✓ gRPC stubs synced to ${_synced} location(s)"
  else
    echo "[geniesim] WARNING: gRPC stub sync skipped (no target dirs found)"
  fi
fi

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

# Enable Isaac Sim's internal ROS 2 libraries for rclpy
# Genie Sim's command_controller.py imports rclpy unconditionally
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

if [ "${GENIESIM_START_SERVER}" = "1" ]; then
  echo "[geniesim] Starting Genie Sim server (logs: ${GENIESIM_SERVER_LOG})"
  # Note: data_collector_server.py doesn't accept --host/--port args
  # It listens on its default port (50051)
  _SERVER_ARGS=""
  [ "${GENIESIM_HEADLESS}" = "1" ] && _SERVER_ARGS="${_SERVER_ARGS} --headless"
  # Only pass --publish_ros when ROS 2 is actually available
  if [ "${GENIESIM_SKIP_ROS_RECORDING:-0}" != "1" ]; then
    _SERVER_ARGS="${_SERVER_ARGS} --publish_ros"
  fi
  nohup "${ISAAC_SIM_PATH}/python.sh" \
    "${GENIESIM_ROOT}/source/data_collection/scripts/data_collector_server.py" \
    ${_SERVER_ARGS} \
    > "${GENIESIM_SERVER_LOG}" 2>&1 &
fi

if [ "${GENIESIM_HEALTHCHECK}" = "1" ]; then
  echo "[geniesim] Running health check (informational only)"
  "${ISAAC_SIM_PATH}/python.sh" -m tools.geniesim_adapter.geniesim_healthcheck || {
    echo "[geniesim] Health check not yet passing - server may still be starting"
  }
fi

# Post-start strict readiness probe (grpc + runtime patch markers + optional
# physics coverage report checks)
if [ "${GENIESIM_START_SERVER}" = "1" ] && [ -f "${SCRIPT_DIR}/readiness_probe.py" ]; then
  _probe_args=(
    --host "${GENIESIM_HOST}"
    --port "${GENIESIM_PORT}"
    --check-patches
    --geniesim-root "${GENIESIM_ROOT}"
    --output "${GENIESIM_RUNTIME_READINESS_JSON}"
  )
  if [ "${GENIESIM_STRICT_RUNTIME_READINESS}" = "1" ]; then
    _probe_args+=(--strict-runtime)
  fi
  echo "[geniesim] Running post-start runtime readiness probe"
  if ! "${ISAAC_SIM_PATH}/python.sh" "${SCRIPT_DIR}/readiness_probe.py" "${_probe_args[@]}"; then
    echo "[geniesim] WARNING: post-start runtime readiness probe failed"
    if [ "${GENIESIM_STRICT_RUNTIME_READINESS}" = "1" ]; then
      if [ "${GENIESIM_STRICT_FAIL_ACTION}" = "error" ]; then
        echo "[geniesim] ERROR: strict runtime readiness requested and post-start probe failed"
        exit 1
      fi
      _mark_force_raw_only "runtime_patch_poststart_probe_failed"
    fi
  fi
fi

# Keep container alive by waiting on the server process
if [ "${GENIESIM_START_SERVER}" = "1" ]; then
  echo "[geniesim] Waiting for server process..."
  # Wait for the server log to be created and then tail it
  while [ ! -f "${GENIESIM_SERVER_LOG}" ]; do sleep 1; done
  tail -f "${GENIESIM_SERVER_LOG}"
fi
