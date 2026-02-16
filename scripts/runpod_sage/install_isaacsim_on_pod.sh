#!/usr/bin/env bash
# =============================================================================
# Install Isaac Sim on the running SAGE + SAM3D pod.
#
# Run this ON THE POD via SSH BEFORE snapshotting.
# This installs Isaac Sim 5.1.0 via pip into a venv and symlinks the SAGE MCP
# extension. After this, the snapshot will include Isaac Sim ready to go.
#
# Usage:
#   bash install_isaacsim_on_pod.sh
#
# Prerequisites:
#   - NVIDIA driver >= 560 (check: nvidia-smi)
#   - Python 3.11 available
#   - SAGE repo at /workspace/SAGE
#   - ~15-20 GB disk space
#
# Takes ~10-20 minutes (pip download + install).
# =============================================================================
set -euo pipefail

log() { echo "[isaacsim-install $(date -u +%FT%TZ)] $*"; }

WORKSPACE=${WORKSPACE:-/workspace}
SAGE_DIR="${WORKSPACE}/SAGE"
VENV_DIR="${WORKSPACE}/isaacsim_env"
BP_DIR="${WORKSPACE}/BlueprintPipeline"
PREFLIGHT_SCRIPT="${BP_DIR}/scripts/runpod_sage/isaacsim_runtime_preflight.sh"
MCP_RESOLVER="${BP_DIR}/scripts/runpod_sage/mcp_extension_paths.py"

resolve_mcp_extension_src() {
    if [[ -f "${MCP_RESOLVER}" ]]; then
        local resolved=""
        resolved="$(python3 "${MCP_RESOLVER}" --sage-dir "${SAGE_DIR}" || true)"
        if [[ -n "${resolved}" ]]; then
            echo "${resolved}"
            return 0
        fi
    fi
    local candidate
    for candidate in \
        "${SAGE_DIR}/server/isaacsim_mcp_ext/isaac.sim.mcp_extension" \
        "${SAGE_DIR}/server/isaacsim/isaac.sim.mcp_extension"; do
        if [[ -d "${candidate}" ]]; then
            if [[ "${candidate}" == "${SAGE_DIR}/server/isaacsim/isaac.sim.mcp_extension" ]]; then
                log "WARNING: Using deprecated MCP extension path: ${candidate}"
            fi
            echo "${candidate}"
            return 0
        fi
    done
    return 1
}

log "=========================================="
log "Isaac Sim 5.1.0 Installation"
log "=========================================="

# ── 1. Check prerequisites ──────────────────────────────────────────────────
log "Checking prerequisites..."

# GPU & driver
if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "ERROR: nvidia-smi not found. GPU required."
    exit 1
fi
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)
log "  GPU: ${GPU_INFO}"

DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [[ "${DRIVER_MAJOR}" -lt 560 ]]; then
    log "ERROR: NVIDIA driver >= 560 required for Isaac Sim 5.1.0."
    log "  Found: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
    exit 2
fi
log "  Driver: OK (>= 560)"

# Python 3.11
if ! command -v python3.11 >/dev/null 2>&1; then
    log "Python 3.11 not found. Installing..."
    apt-get update -qq
    apt-get install -y -qq software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update -qq
    apt-get install -y -qq python3.11 python3.11-venv python3.11-dev
fi
log "  Python 3.11: $(python3.11 --version)"

# Disk space
AVAIL_GB=$(df -BG "${WORKSPACE}" | tail -1 | awk '{print $4}' | tr -d 'G')
log "  Available disk: ${AVAIL_GB}G"
if [[ "${AVAIL_GB}" -lt 15 ]]; then
    log "WARNING: Less than 15GB free. Isaac Sim needs ~12-15GB."
    log "  Cleaning up to free space..."
    pip cache purge 2>/dev/null || true
    conda clean -afy 2>/dev/null || true
    find /workspace -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
    AVAIL_GB=$(df -BG "${WORKSPACE}" | tail -1 | awk '{print $4}' | tr -d 'G')
    log "  Available after cleanup: ${AVAIL_GB}G"
fi

# System deps for Isaac Sim
log "Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq \
    libgl1 libglib2.0-0 libxrender1 libxi6 libxxf86vm1 \
    libxfixes3 libxkbcommon0 libsm6 libice6 libxext6 \
    libxrandr2 libxcursor1 libxinerama1 libepoxy0 libxt6 \
    libglu1-mesa libegl1 libegl-mesa0 libgles2-mesa \
    libopengl0 libglx-mesa0 libvulkan1 vulkan-tools \
    psmisc iproute2 >/dev/null
log "  System deps: OK"

if [[ -x "${PREFLIGHT_SCRIPT}" ]]; then
    log "Running Isaac Sim runtime preflight (Vulkan + import safety)..."
    ISAACSIM_PY="${VENV_DIR}/bin/python3" REQUIRE_LOCAL_ROBOT_ASSET="${REQUIRE_LOCAL_ROBOT_ASSET:-0}" "${PREFLIGHT_SCRIPT}"
fi
if [[ -z "${VK_ICD_FILENAMES:-}" && -z "${VK_DRIVER_FILES:-}" ]]; then
    for candidate in /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json; do
        if [[ -f "${candidate}" ]]; then
            export VK_ICD_FILENAMES="${candidate}"
            export VK_DRIVER_FILES="${candidate}"
            log "Using NVIDIA Vulkan ICD: ${candidate}"
            break
        fi
    done
fi
if [[ -n "${VK_ICD_FILENAMES:-}" && -z "${VK_DRIVER_FILES:-}" ]]; then
    export VK_DRIVER_FILES="${VK_ICD_FILENAMES}"
fi
if [[ -n "${VK_DRIVER_FILES:-}" && -z "${VK_ICD_FILENAMES:-}" ]]; then
    export VK_ICD_FILENAMES="${VK_DRIVER_FILES}"
fi

# ── 2. Create venv and install Isaac Sim ─────────────────────────────────────
if [[ -d "${VENV_DIR}" ]]; then
    log "Isaac Sim venv already exists at ${VENV_DIR}"
    # Check if Isaac Sim is actually importable
    if "${VENV_DIR}/bin/python3" -c "import isaacsim" 2>/dev/null; then
        log "  Isaac Sim already installed and importable!"
        SKIP_INSTALL=1
    else
        log "  Venv exists but Isaac Sim not importable. Reinstalling..."
        SKIP_INSTALL=0
    fi
else
    SKIP_INSTALL=0
fi

if [[ "${SKIP_INSTALL}" == "0" ]]; then
    log "Creating Python 3.11 venv at ${VENV_DIR}..."
    python3.11 -m venv "${VENV_DIR}"

    log "Upgrading pip..."
    "${VENV_DIR}/bin/pip" install --upgrade pip -q

    log "Installing Isaac Sim 5.1.0 via pip..."
    log "  (This takes 10-20 minutes — downloading ~8-12 GB)"
    "${VENV_DIR}/bin/pip" install --no-cache-dir \
        "isaacsim[all,extscache]==5.1.0" \
        --extra-index-url https://pypi.nvidia.com 2>&1 | tail -10

    log "Isaac Sim installed."
fi

# ── 3. Get Isaac Sim path and symlink MCP extension ─────────────────────────
log "Resolving Isaac Sim path..."
ISAACSIM_PATH=$("${VENV_DIR}/bin/python3" -P -c \
    "import os, pathlib, sys; blocked=pathlib.Path('${SAGE_DIR}/server').resolve(); sys.path=[p for p in sys.path if p and pathlib.Path(p).resolve()!=blocked]; import isaacsim; print(os.path.dirname(isaacsim.__file__))")
log "  Path: ${ISAACSIM_PATH}"

# Save path for entrypoint
echo "ISAACSIM_PATH=${ISAACSIM_PATH}" > "${WORKSPACE}/.isaacsim_path"

# Symlink SAGE MCP extension
MCP_EXT_SRC="$(resolve_mcp_extension_src || true)"
MCP_EXT_DST="${ISAACSIM_PATH}/exts/isaac.sim.mcp_extension"

if [[ -d "${MCP_EXT_SRC}" ]]; then
    log "Symlinking SAGE MCP extension..."
    ln -sf "${MCP_EXT_SRC}" "${MCP_EXT_DST}"
    log "  ${MCP_EXT_SRC} → ${MCP_EXT_DST}"
else
    log "WARNING: SAGE MCP extension not found at ${MCP_EXT_SRC}"
    log "  Isaac Sim will run but without MCP integration."
fi

# ── 4. Find kit file ────────────────────────────────────────────────────────
log "Finding .kit experience file..."
KIT_FILE=""
for candidate in \
    "${ISAACSIM_PATH}/apps/omni.isaac.sim.kit" \
    "${ISAACSIM_PATH}/apps/isaacsim.exp.full.kit" \
    "${ISAACSIM_PATH}/apps/isaacsim.exp.base.kit"; do
    if [[ -f "${candidate}" ]]; then
        KIT_FILE="${candidate}"
        break
    fi
done

if [[ -z "${KIT_FILE}" ]]; then
    KIT_FILE=$(find "${ISAACSIM_PATH}/apps" -name "*.kit" -print -quit 2>/dev/null || true)
fi

if [[ -n "${KIT_FILE}" ]]; then
    log "  Kit file: ${KIT_FILE}"
    # Save for entrypoint
    echo "KIT_FILE=${KIT_FILE}" >> "${WORKSPACE}/.isaacsim_path"
else
    log "  WARNING: No .kit file found (will search at startup)"
fi

# ── 5. Verify install ───────────────────────────────────────────────────────
log ""
log "Verifying installation..."
"${VENV_DIR}/bin/python3" -P -c "
import isaacsim
import os
path = os.path.dirname(isaacsim.__file__)
print(f'  isaacsim: OK (at {path})')
kit = os.path.join(path, 'kit', 'kit')
print(f'  kit binary: {\"OK\" if os.path.exists(kit) else \"MISSING\"} ({kit})')
exts = os.path.join(path, 'exts')
mcp = os.path.join(exts, 'isaac.sim.mcp_extension')
print(f'  MCP extension: {\"OK\" if os.path.exists(mcp) else \"MISSING\"} ({mcp})')
"

INSTALL_SIZE=$(du -sh "${VENV_DIR}" 2>/dev/null | cut -f1)
log ""
log "=========================================="
log "Isaac Sim installation complete!"
log "=========================================="
log ""
log "  Install size: ${INSTALL_SIZE}"
log "  Path: ${ISAACSIM_PATH}"
log "  Kit file: ${KIT_FILE:-searching at startup}"
log ""
log "To start Isaac Sim manually:"
log "  source ${VENV_DIR}/bin/activate"
log "  export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y"
log "  export SLURM_JOB_ID=\${SLURM_JOB_ID:-12345}"
log "  ${ISAACSIM_PATH}/kit/kit ${KIT_FILE} --no-window --enable isaac.sim.mcp_extension"
log ""
log "Or it will start automatically via entrypoint.sh on next container boot."
log ""
log "Now run capture_pod_snapshot.sh to create the full snapshot."
