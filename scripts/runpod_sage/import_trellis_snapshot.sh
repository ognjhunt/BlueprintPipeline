#!/usr/bin/env bash
# import_trellis_snapshot.sh — Restore a TRELLIS snapshot on a new machine.
#
# Usage:
#   bash import_trellis_snapshot.sh /path/to/trellis_snapshot_YYYYMMDD.tar.gz
#
# Prerequisites on the target machine:
#   - NVIDIA GPU with driver >= 535
#   - Linux x86_64
#   - ~20GB free disk space
#   - libx11-6, libgl1 (will attempt to install if missing)
#
# After import, start the server with:
#   bash run_trellis.sh
set -euo pipefail

log() { echo "[import-snapshot $(date -u +%FT%TZ)] $*"; }

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <snapshot_tar_gz_path>"
    echo ""
    echo "Example:"
    echo "  $0 /workspace/trellis_snapshot_20260214.tar.gz"
    exit 1
fi

SNAPSHOT_PATH="$1"

if [[ ! -f "${SNAPSHOT_PATH}" ]]; then
    log "ERROR: Snapshot file not found: ${SNAPSHOT_PATH}"
    exit 2
fi

# ──────────────────────────────────────────────
# 1. Verify checksum (if available)
# ──────────────────────────────────────────────
if [[ -f "${SNAPSHOT_PATH}.sha256" ]]; then
    log "Verifying checksum..."
    cd "$(dirname ${SNAPSHOT_PATH})"
    if sha256sum -c "${SNAPSHOT_PATH}.sha256" 2>/dev/null; then
        log "Checksum OK"
    else
        log "WARNING: Checksum mismatch! Archive may be corrupted."
        read -p "Continue anyway? [y/N] " -n 1 -r
        echo
        [[ $REPLY =~ ^[Yy]$ ]] || exit 3
    fi
else
    log "No checksum file found — skipping verification."
fi

# ──────────────────────────────────────────────
# 2. Check GPU
# ──────────────────────────────────────────────
log "Checking GPU..."
if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "ERROR: nvidia-smi not found. NVIDIA GPU driver required."
    exit 4
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1

# ──────────────────────────────────────────────
# 3. Install system dependencies
# ──────────────────────────────────────────────
log "Ensuring system dependencies..."
apt-get update -qq 2>/dev/null && apt-get install -y -qq libx11-6 libgl1 curl >/dev/null 2>&1 || {
    log "WARNING: Could not install system packages. If libx11/libgl1 are missing, nvdiffrast may fail."
}

# ──────────────────────────────────────────────
# 4. Extract snapshot
# ──────────────────────────────────────────────
log "Extracting snapshot to / (this may take 5-10 minutes)..."
SNAPSHOT_SIZE=$(du -sh "${SNAPSHOT_PATH}" | cut -f1)
log "  Archive size: ${SNAPSHOT_SIZE}"

# Check disk space
DISK_FREE_MB=$(df -m / | tail -1 | awk '{print $4}')
ARCHIVE_MB=$(du -m "${SNAPSHOT_PATH}" | cut -f1)
NEEDED_MB=$(( ARCHIVE_MB * 2 ))  # rough: compressed * 2
if [[ "${DISK_FREE_MB}" -lt "${NEEDED_MB}" ]]; then
    log "WARNING: Only ${DISK_FREE_MB}MB free, may need ~${NEEDED_MB}MB."
fi

tar xzf "${SNAPSHOT_PATH}" -C / 2>/dev/null || {
    log "ERROR: tar extraction failed."
    exit 5
}

# ──────────────────────────────────────────────
# 5. Verify extraction
# ──────────────────────────────────────────────
log "Verifying extracted files..."
ERRORS=0

check_exists() {
    if [[ ! -e "$1" ]]; then
        log "  MISSING: $1"
        ERRORS=$((ERRORS + 1))
    else
        log "  OK: $1"
    fi
}

check_exists /root/miniconda3/envs/trellis/bin/python
check_exists /root/.cache/huggingface/hub/models--microsoft--TRELLIS-text-xlarge
check_exists /workspace/SAGE/server/TRELLIS/server.py
check_exists /workspace/SAGE/server/TRELLIS/central_server.py

if [[ "${ERRORS}" -gt 0 ]]; then
    log "ERROR: ${ERRORS} expected files missing after extraction."
    exit 6
fi

# ──────────────────────────────────────────────
# 6. Re-initialize conda
# ──────────────────────────────────────────────
log "Re-initializing conda..."
/root/miniconda3/bin/conda init bash 2>/dev/null || true

# ──────────────────────────────────────────────
# 7. Quick import test
# ──────────────────────────────────────────────
log "Running quick import test..."
source /root/miniconda3/etc/profile.d/conda.sh
conda activate trellis

python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')
import nvdiffrast
print('nvdiffrast OK')
import flask
print('Flask OK')
from trellis.pipelines import TrellisTextTo3DPipeline
print('TRELLIS pipeline importable OK')
print('All checks passed!')
" 2>/dev/null || {
    log "WARNING: Some imports failed. You may need to rebuild nvdiffrast:"
    log "  pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation"
    log "  pip install 'git+https://github.com/autonomousvision/mip-splatting.git#subdirectory=submodules/diff-gaussian-rasterization/' --no-build-isolation"
}

# ──────────────────────────────────────────────
# 8. Done
# ──────────────────────────────────────────────
log ""
log "=== IMPORT COMPLETE ==="
log "To start the TRELLIS server:"
log "  bash run_trellis.sh"
log ""
log "Or manually:"
log "  source /root/miniconda3/etc/profile.d/conda.sh"
log "  conda activate trellis"
log "  cd /workspace/SAGE/server/TRELLIS"
log "  python server.py --port 8081 --gpu 0"
