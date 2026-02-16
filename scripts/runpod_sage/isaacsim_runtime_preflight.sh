#!/usr/bin/env bash
# =============================================================================
# Isaac Sim runtime preflight (Vulkan + Python import safety).
#
# Usage:
#   bash isaacsim_runtime_preflight.sh            # strict (exit non-zero on fail)
#   bash isaacsim_runtime_preflight.sh --warn-only
#
# Optional env:
#   ISAACSIM_PY    Python used for isaacsim import check (default: /workspace/isaacsim_env/bin/python3)
#   SAGE_SERVER_DIR  Path that may shadow isaacsim module (default: /workspace/SAGE/server)
# =============================================================================
set -euo pipefail

log() { echo "[isaacsim-preflight $(date -u +%FT%TZ)] $*"; }

STRICT=1
if [[ "${1:-}" == "--warn-only" ]]; then
    STRICT=0
fi

ISAACSIM_PY="${ISAACSIM_PY:-/workspace/isaacsim_env/bin/python3}"
SAGE_SERVER_DIR="${SAGE_SERVER_DIR:-/workspace/SAGE/server}"
ISAAC_ASSETS_ROOT="${ISAAC_ASSETS_ROOT:-/workspace/isaacsim_assets/Assets/Isaac/5.1}"
REQUIRE_LOCAL_ROBOT_ASSET="${REQUIRE_LOCAL_ROBOT_ASSET:-1}"
BP_DIR="${BP_DIR:-/workspace/BlueprintPipeline}"
ASSET_VALIDATOR="${ASSET_VALIDATOR:-${BP_DIR}/scripts/runpod_sage/validate_ridgeback_usd_assets.py}"

require_cmd() {
    local cmd="$1" help_msg="$2"
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        log "ERROR: Missing command '${cmd}'. ${help_msg}"
        return 1
    fi
    return 0
}

run_checks() {
    local driver_version driver_major
    local -a missing_libs
    local tmp icd

    require_cmd nvidia-smi "A GPU-enabled runtime is required." || return 1

    driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 || true)"
    driver_major="$(echo "${driver_version}" | cut -d. -f1)"
    if [[ -z "${driver_major}" || ! "${driver_major}" =~ ^[0-9]+$ ]]; then
        log "ERROR: Could not parse NVIDIA driver version from nvidia-smi output: '${driver_version}'"
        return 1
    fi
    if [[ "${driver_major}" -lt 560 ]]; then
        log "ERROR: NVIDIA driver >= 560 required for Isaac Sim 5.1.0. Found: ${driver_version}"
        return 1
    fi
    log "NVIDIA driver OK: ${driver_version}"

    missing_libs=()
    for lib in libvulkan.so.1 libGLU.so.1 libXt.so.6; do
        if ! ldconfig -p 2>/dev/null | grep -q "${lib}"; then
            missing_libs+=("${lib}")
        fi
    done
    if [[ "${#missing_libs[@]}" -gt 0 ]]; then
        log "ERROR: Missing runtime libraries: ${missing_libs[*]}"
        log "Install: apt-get update && apt-get install -y libvulkan1 libglu1-mesa libxt6 vulkan-tools"
        return 1
    fi
    log "Shared libraries OK: libvulkan.so.1 libGLU.so.1 libXt.so.6"

    require_cmd vulkaninfo "Install 'vulkan-tools' and ensure host Vulkan ICD passthrough is enabled." || return 1

    tmp="$(mktemp /tmp/isaacsim_vulkaninfo.XXXXXX.log)"
    icd=""
    if [[ -z "${VK_ICD_FILENAMES:-}" && -z "${VK_DRIVER_FILES:-}" ]]; then
        for candidate in /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json; do
            if [[ -f "${candidate}" ]]; then
                icd="${candidate}"
                break
            fi
        done
    fi

    if [[ -n "${icd}" ]]; then
        if ! VK_ICD_FILENAMES="${icd}" VK_DRIVER_FILES="${icd}" vulkaninfo --summary >"${tmp}" 2>&1; then
            log "ERROR: vulkaninfo failed with VK_ICD_FILENAMES/VK_DRIVER_FILES=${icd}."
            tail -n 40 "${tmp}" || true
            rm -f "${tmp}"
            return 1
        fi
    else
        if ! vulkaninfo --summary >"${tmp}" 2>&1; then
            log "ERROR: vulkaninfo failed."
            tail -n 40 "${tmp}" || true
            rm -f "${tmp}"
            return 1
        fi
    fi

    if grep -q "ERROR_INCOMPATIBLE_DRIVER" "${tmp}" || grep -qi "Failed to create Vulkan instance" "${tmp}"; then
        log "ERROR: Vulkan ICD/driver mismatch detected."
        tail -n 40 "${tmp}" || true
        log "Likely cause: host NVIDIA Vulkan runtime is not correctly passed through to the container."
        log "Confirm container runs with NVIDIA runtime and graphics capability (e.g. --gpus all, NVIDIA_DRIVER_CAPABILITIES includes graphics)."
        rm -f "${tmp}"
        return 1
    fi
    log "Vulkan preflight OK."
    rm -f "${tmp}"

    if [[ -x "${ISAACSIM_PY}" ]]; then
        if ! SAGE_SERVER_DIR="${SAGE_SERVER_DIR}" "${ISAACSIM_PY}" -P - <<'PY' >/tmp/isaacsim_import_check.log 2>&1
import os
import pathlib
import sys

blocked = pathlib.Path(os.environ.get("SAGE_SERVER_DIR", "/workspace/SAGE/server")).resolve()
cleaned = []
for p in sys.path:
    if not p:
        continue
    try:
        rp = pathlib.Path(p).resolve()
    except Exception:
        cleaned.append(p)
        continue
    if rp == blocked:
        continue
    cleaned.append(p)
sys.path[:] = cleaned

import isaacsim  # noqa: F401

mod_path = pathlib.Path(isaacsim.__file__).resolve()
if str(mod_path).startswith(str(blocked) + os.sep):
    raise RuntimeError(f"isaacsim import shadowed by local path: {mod_path}")

print(mod_path)
PY
        then
            log "ERROR: Isaac Sim Python import check failed."
            tail -n 60 /tmp/isaacsim_import_check.log || true
            return 1
        fi
        log "Isaac Sim import preflight OK: $(tail -n 1 /tmp/isaacsim_import_check.log)"
    else
        log "Isaac Sim Python not found at ${ISAACSIM_PY}; skipping import check."
    fi

    local local_robot_usd
    local_robot_usd="${ISAAC_ASSETS_ROOT}/Isaac/Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd"
    if [[ -f "${local_robot_usd}" ]]; then
        log "Local RidgebackFranka asset found: ${local_robot_usd}"
        if [[ -f "${ASSET_VALIDATOR}" ]]; then
            local report_path
            report_path="/tmp/ridgeback_asset_validation.json"
            if ! python3 "${ASSET_VALIDATOR}" \
                --root-usd "${local_robot_usd}" \
                --assets-root "${ISAAC_ASSETS_ROOT}" \
                --report-path "${report_path}"; then
                log "ERROR: RidgebackFranka transitive asset validation failed."
                tail -n 40 "${report_path}" 2>/dev/null || true
                return 1
            fi
            log "RidgebackFranka transitive asset validation OK."
        else
            log "WARNING: Asset validator script not found: ${ASSET_VALIDATOR}"
        fi
    else
        if [[ "${REQUIRE_LOCAL_ROBOT_ASSET}" == "1" ]]; then
            log "ERROR: Local RidgebackFranka asset missing: ${local_robot_usd}"
            return 1
        fi
        log "WARNING: Local RidgebackFranka asset not found; Stage 7 may fetch /Isaac assets at runtime."
    fi

    return 0
}

if run_checks; then
    log "Preflight PASSED."
    exit 0
fi

if [[ "${STRICT}" == "1" ]]; then
    log "Preflight FAILED."
    exit 1
fi

log "WARNING: Preflight failed, continuing due to --warn-only."
exit 0
