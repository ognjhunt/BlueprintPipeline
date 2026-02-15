#!/usr/bin/env bash
# =============================================================================
# Install Interactive Asset Backends into the SAGE Pod
#
# Run this ONCE on the Vast.ai pod, then take a snapshot. After that, every
# new pod start has everything baked in — no runtime downloads.
#
# What this installs:
#   1. pybullet + trimesh (into sage conda env) — PyBullet critic
#   2. PhysX-Anything repo + VLM weights (~4GB)  — image → URDF backend
#   3. Infinigen repo + pip install              — category → URDF backend
#   4. Local HTTP wrappers for both backends (run as localhost services)
#
# Usage (on the pod via SSH):
#   bash /workspace/BlueprintPipeline/scripts/runpod_sage/install_interactive_backends.sh
#
# After running:
#   - Take a Vast.ai snapshot (or run capture_pod_snapshot.sh)
#   - Future pods start with everything ready
#
# Disk usage: ~6-8 GB total
# VRAM usage at runtime: ~4GB PhysX-Anything VLM (loaded on demand, not at boot)
# =============================================================================
set -euo pipefail

log() { echo "[install-backends $(date -u +%FT%TZ)] $*"; }

WORKSPACE="${WORKSPACE:-/workspace}"
CONDA_DIR="${WORKSPACE}/miniconda3"
SAGE_ENV="sage"
BP_DIR="${WORKSPACE}/BlueprintPipeline"

PHYSX_ANYTHING_DIR="${WORKSPACE}/PhysX-Anything"
INFINIGEN_DIR="${WORKSPACE}/infinigen"

# ── 0. Preflight checks ─────────────────────────────────────────────────────
log "=========================================="
log "Interactive Backend Installer"
log "=========================================="

if [[ ! -d "${CONDA_DIR}" ]]; then
    log "ERROR: Conda not found at ${CONDA_DIR}"
    exit 1
fi

# shellcheck disable=SC1091
source "${CONDA_DIR}/etc/profile.d/conda.sh"
conda activate "${SAGE_ENV}" 2>/dev/null || {
    log "ERROR: Cannot activate conda env '${SAGE_ENV}'"
    exit 1
}

log "Python: $(which python) ($(python --version))"
log "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not installed')"
log "CUDA: $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'N/A')"

DISK_AVAIL=$(df -BG "${WORKSPACE}" 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G')
log "Disk available: ${DISK_AVAIL}G"
if [[ -n "${DISK_AVAIL}" ]] && [[ "${DISK_AVAIL}" -lt 10 ]]; then
    log "WARNING: Less than 10GB free. May not have enough space."
fi

# ── 1. Install Python packages into sage env ─────────────────────────────────
log ""
log "════ Step 1/4: Python packages (pybullet, trimesh, flask) ════"

# Check what's already installed
NEED_INSTALL=""
for pkg in pybullet trimesh flask gunicorn; do
    if ! python -c "import ${pkg}" 2>/dev/null; then
        NEED_INSTALL="${NEED_INSTALL} ${pkg}"
    else
        log "  Already installed: ${pkg}"
    fi
done

# pyglet is needed for trimesh offscreen rendering
if ! python -c "import pyglet" 2>/dev/null; then
    NEED_INSTALL="${NEED_INSTALL} pyglet"
fi

# gdown is needed for PhysX-Anything model downloads
if ! python -c "import gdown" 2>/dev/null; then
    NEED_INSTALL="${NEED_INSTALL} gdown"
fi

if [[ -n "${NEED_INSTALL}" ]]; then
    log "Installing:${NEED_INSTALL}"
    pip install --no-cache-dir ${NEED_INSTALL} "trimesh[easy]"
else
    log "All Python packages already installed."
fi

# ── 2. PhysX-Anything ────────────────────────────────────────────────────────
log ""
log "════ Step 2/4: PhysX-Anything (image → URDF) ════"

if [[ -d "${PHYSX_ANYTHING_DIR}" ]] && [[ -d "${PHYSX_ANYTHING_DIR}/pretrain/vlm" ]]; then
    log "PhysX-Anything already installed at ${PHYSX_ANYTHING_DIR}"
    log "  Checkpoint: $(du -sh "${PHYSX_ANYTHING_DIR}/pretrain/vlm" 2>/dev/null | cut -f1)"
else
    log "Cloning PhysX-Anything..."
    if [[ -d "${PHYSX_ANYTHING_DIR}" ]]; then
        log "  Repo exists but weights missing; downloading weights..."
    else
        git clone https://github.com/ziangcao0312/PhysX-Anything.git "${PHYSX_ANYTHING_DIR}"
    fi

    log "Installing PhysX-Anything Python dependencies..."
    # Filter out torch/torchvision (already in sage env with correct CUDA version)
    if [[ -f "${PHYSX_ANYTHING_DIR}/requirements.txt" ]]; then
        python -c "
import pathlib
p = pathlib.Path('${PHYSX_ANYTHING_DIR}/requirements.txt')
lines = p.read_text().splitlines()
filtered = [l for l in lines if not l.strip().startswith(('torch==', 'torchvision==', 'torch ', 'torchvision '))]
pathlib.Path('/tmp/physx_requirements_filtered.txt').write_text('\n'.join(filtered) + '\n')
"
        pip install --no-cache-dir -r /tmp/physx_requirements_filtered.txt 2>&1 | tail -5
        rm -f /tmp/physx_requirements_filtered.txt
    fi

    log "Downloading PhysX-Anything model weights (this takes a few minutes)..."
    cd "${PHYSX_ANYTHING_DIR}"
    python download.py
    cd "${WORKSPACE}"

    CKPT_SIZE=$(du -sh "${PHYSX_ANYTHING_DIR}/pretrain/vlm" 2>/dev/null | cut -f1 || echo "unknown")
    log "PhysX-Anything installed. Checkpoint: ${CKPT_SIZE}"
fi

# ── 3. Infinigen ─────────────────────────────────────────────────────────────
log ""
log "════ Step 3/4: Infinigen (category → URDF) ════"

if [[ -d "${INFINIGEN_DIR}" ]]; then
    log "Infinigen already installed at ${INFINIGEN_DIR}"
    # Verify it's importable
    if python -c "import infinigen" 2>/dev/null; then
        log "  Infinigen is importable."
    else
        log "  Infinigen exists but not importable. Running pip install -e..."
        pip install --no-cache-dir -e "${INFINIGEN_DIR}" 2>&1 | tail -5
    fi
else
    log "Cloning Infinigen..."
    git clone https://github.com/princeton-vl/infinigen.git "${INFINIGEN_DIR}"

    log "Installing Infinigen (editable)..."
    pip install --no-cache-dir -e "${INFINIGEN_DIR}" 2>&1 | tail -5

    if python -c "import infinigen" 2>/dev/null; then
        log "Infinigen installed and importable."
    else
        log "WARNING: Infinigen installed but not importable (may need additional system deps)."
        log "  The service wrapper will still work if scripts/spawn_asset.py runs standalone."
    fi
fi

# ── 4. Copy service wrappers ─────────────────────────────────────────────────
log ""
log "════ Step 4/4: Service wrappers ════"

# Copy the lightweight HTTP service wrappers to /workspace so they're
# available outside the git repo (and survive repo updates).
for svc in infinigen_service.py physx_anything_service.py; do
    SRC_DIR=""
    if [[ "${svc}" == "infinigen_service.py" ]]; then
        SRC_DIR="${BP_DIR}/infinigen-service"
    else
        SRC_DIR="${BP_DIR}/physx-anything-service"
    fi

    if [[ -f "${SRC_DIR}/${svc}" ]]; then
        cp -f "${SRC_DIR}/${svc}" "${WORKSPACE}/${svc}"
        log "  Copied ${svc} → /workspace/${svc}"
    else
        log "  WARNING: ${SRC_DIR}/${svc} not found"
    fi
done

# ── 5. Verify installation ───────────────────────────────────────────────────
log ""
log "════ Verification ════"

PASS=0
FAIL=0

check() {
    local desc="$1" cmd="$2"
    if eval "${cmd}" >/dev/null 2>&1; then
        log "  PASS: ${desc}"
        PASS=$((PASS + 1))
    else
        log "  FAIL: ${desc}"
        FAIL=$((FAIL + 1))
    fi
}

check "pybullet importable"          "python -c 'import pybullet'"
check "trimesh importable"           "python -c 'import trimesh'"
check "flask importable"             "python -c 'import flask'"
check "PhysX-Anything repo"          "test -f '${PHYSX_ANYTHING_DIR}/1_vlm_demo.py'"
check "PhysX-Anything weights"       "test -d '${PHYSX_ANYTHING_DIR}/pretrain/vlm'"
check "Infinigen repo"               "test -f '${INFINIGEN_DIR}/scripts/spawn_asset.py'"
check "infinigen_service.py"         "test -f '${WORKSPACE}/infinigen_service.py'"
check "physx_anything_service.py"    "test -f '${WORKSPACE}/physx_anything_service.py'"

log ""
log "=========================================="
log "Installation complete: ${PASS} passed, ${FAIL} failed"
log "=========================================="
log ""
log "Disk usage:"
log "  PhysX-Anything: $(du -sh "${PHYSX_ANYTHING_DIR}" 2>/dev/null | cut -f1 || echo 'N/A')"
log "  Infinigen:      $(du -sh "${INFINIGEN_DIR}" 2>/dev/null | cut -f1 || echo 'N/A')"
log ""
log "Next steps:"
log "  1. Test the backends:"
log "     bash ${BP_DIR}/scripts/runpod_sage/start_interactive_backends.sh"
log ""
log "  2. Take a snapshot so new pods have everything baked in:"
log "     bash ${BP_DIR}/scripts/runpod_sage/capture_pod_snapshot.sh"
log ""
log "  3. Run the full pipeline with articulation:"
log "     ARTICULATION_BACKEND=auto bash ${BP_DIR}/scripts/runpod_sage/run_full_pipeline.sh"
log "=========================================="
