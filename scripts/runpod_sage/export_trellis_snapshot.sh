#!/usr/bin/env bash
# export_trellis_snapshot.sh — Create a portable tar archive of everything
# needed to run TRELLIS on any Linux machine with an NVIDIA GPU.
#
# Run this ON the pod (Pod A) after a successful bootstrap.
#
# What it packages (~15GB compressed):
#   - /root/miniconda3/                              (~10GB) conda + trellis env
#   - /root/.cache/huggingface/hub/models--microsoft* (~3.9GB) model weights
#   - /workspace/SAGE/server/TRELLIS/                 (~1.2GB) repo + server scripts
#   - /workspace/SAGE/server/*.py (key server files)
#   - Metadata (package list, GPU info, export date)
#
# Output: /workspace/trellis_snapshot_YYYYMMDD.tar.gz
#         (or custom path via SNAPSHOT_PATH env var)
#
# To restore: see import_trellis_snapshot.sh
set -euo pipefail

log() { echo "[export-snapshot $(date -u +%FT%TZ)] $*"; }

DATE_TAG=$(date -u +%Y%m%d_%H%M%S)
SNAPSHOT_PATH=${SNAPSHOT_PATH:-/workspace/trellis_snapshot_${DATE_TAG}.tar.gz}
MANIFEST_PATH="/tmp/trellis_snapshot_manifest_${DATE_TAG}.json"

log "Creating TRELLIS portable snapshot"
log "Output: ${SNAPSHOT_PATH}"

# ──────────────────────────────────────────────
# 1. Generate manifest (for verification on import)
# ──────────────────────────────────────────────
log "Generating manifest..."
python3 -c "
import json, subprocess, os, datetime

manifest = {
    'export_date': '${DATE_TAG}',
    'hostname': os.uname().nodename,
    'gpu': subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
        text=True
    ).strip(),
    'cuda_version': subprocess.check_output(
        ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
        text=True
    ).strip(),
    'python_version': subprocess.check_output(['python3', '--version'], text=True).strip(),
    'contents': {
        'miniconda3': '/root/miniconda3/',
        'hf_cache': '/root/.cache/huggingface/hub/models--microsoft--TRELLIS-text-xlarge/',
        'trellis_repo': '/workspace/SAGE/server/TRELLIS/',
        'sage_server_scripts': '/workspace/SAGE/server/'
    },
    'restore_script': 'import_trellis_snapshot.sh',
    'run_script': 'run_trellis.sh'
}

# Get pip package list from trellis env
try:
    pkgs = subprocess.check_output(
        ['/root/miniconda3/envs/trellis/bin/pip', 'list', '--format=json'],
        text=True
    )
    manifest['packages'] = json.loads(pkgs)
except:
    manifest['packages'] = []

with open('${MANIFEST_PATH}', 'w') as f:
    json.dump(manifest, f, indent=2)
print(f'Manifest written to ${MANIFEST_PATH}')
"

# ──────────────────────────────────────────────
# 2. Estimate size
# ──────────────────────────────────────────────
log "Estimating snapshot size..."
CONDA_SIZE=$(du -sm /root/miniconda3 2>/dev/null | cut -f1)
HF_SIZE=$(du -sm /root/.cache/huggingface/hub/models--microsoft--TRELLIS-text-xlarge 2>/dev/null | cut -f1 || echo 0)
TRELLIS_SIZE=$(du -sm /workspace/SAGE/server/TRELLIS 2>/dev/null | cut -f1)
TOTAL_MB=$(( CONDA_SIZE + HF_SIZE + TRELLIS_SIZE ))
log "  Conda env:     ${CONDA_SIZE}MB"
log "  HF models:     ${HF_SIZE}MB"
log "  TRELLIS repo:  ${TRELLIS_SIZE}MB"
log "  Total (uncompressed): ~${TOTAL_MB}MB"
log "  Estimated compressed: ~$(( TOTAL_MB * 60 / 100 ))MB"

DISK_FREE=$(df -m /workspace | tail -1 | awk '{print $4}')
NEEDED=$(( TOTAL_MB * 60 / 100 + 500 ))  # compressed + margin
if [[ "${DISK_FREE}" -lt "${NEEDED}" ]]; then
    log "WARNING: Only ${DISK_FREE}MB free, need ~${NEEDED}MB for snapshot."
    log "Consider using SNAPSHOT_PATH pointing to an external mount."
fi

# ──────────────────────────────────────────────
# 3. Create tar archive
# ──────────────────────────────────────────────
log "Creating tar archive (this may take 10-20 minutes)..."

# We use relative paths from / so the archive extracts cleanly
tar czf "${SNAPSHOT_PATH}" \
    --warning=no-file-changed \
    -C / \
    root/miniconda3 \
    root/.cache/huggingface/hub/models--microsoft--TRELLIS-text-xlarge \
    workspace/SAGE/server/TRELLIS \
    workspace/SAGE/server/server.py \
    workspace/SAGE/server/central_server.py \
    -C /tmp \
    "$(basename ${MANIFEST_PATH})" \
    2>/dev/null || {
        # tar exits 1 on "file changed as we read it" — that's fine
        if [[ $? -gt 1 ]]; then
            log "ERROR: tar failed."
            exit 1
        fi
    }

SNAPSHOT_SIZE=$(du -sh "${SNAPSHOT_PATH}" | cut -f1)
log "Snapshot created: ${SNAPSHOT_PATH} (${SNAPSHOT_SIZE})"

# ──────────────────────────────────────────────
# 4. Generate checksums
# ──────────────────────────────────────────────
log "Generating checksum..."
SHA256=$(sha256sum "${SNAPSHOT_PATH}" | cut -d' ' -f1)
echo "${SHA256}  $(basename ${SNAPSHOT_PATH})" > "${SNAPSHOT_PATH}.sha256"
log "SHA256: ${SHA256}"

log ""
log "=== EXPORT COMPLETE ==="
log "Snapshot:  ${SNAPSHOT_PATH}"
log "Checksum:  ${SNAPSHOT_PATH}.sha256"
log "Manifest:  ${MANIFEST_PATH}"
log ""
log "To download to your local machine:"
log "  scp -P <PORT> -i ~/.ssh/id_ed25519 root@<IP>:${SNAPSHOT_PATH} ."
log ""
log "To upload to a new provider:"
log "  scp -P <PORT> -i <KEY> ${SNAPSHOT_PATH} root@<IP>:/workspace/"
log "  # Then on the new machine:"
log "  bash import_trellis_snapshot.sh /workspace/$(basename ${SNAPSHOT_PATH})"
