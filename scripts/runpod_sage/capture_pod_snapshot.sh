#!/usr/bin/env bash
# =============================================================================
# Capture a complete snapshot of the running SAGE + SAM3D pod.
#
# Run this ON THE POD via SSH after your SAGE run completes.
# It creates a portable tar.gz archive that can be imported on any machine.
#
# Usage:
#   bash capture_pod_snapshot.sh
#
# Output:
#   /workspace/sage-sam3d-snapshot-YYYYMMDD.tar.gz  (~20-30 GB compressed)
#
# To import on a new machine/pod:
#   tar xzf sage-sam3d-snapshot-*.tar.gz -C /
#   bash /workspace/entrypoint.sh
# =============================================================================
set -euo pipefail

log() { echo "[capture $(date -u +%FT%TZ)] $*"; }

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST="/workspace/snapshot_manifest_${TIMESTAMP}.json"
ARCHIVE="/workspace/sage-sam3d-snapshot-${TIMESTAMP}.tar.gz"

log "=========================================="
log "SAGE + SAM3D Pod Snapshot"
log "=========================================="

# ── 1. Inventory what's installed ────────────────────────────────────────────
log "Creating manifest..."

python3 - > "${MANIFEST}" <<'PYEOF'
import json, os, subprocess, datetime, shutil

def run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL, timeout=10).decode().strip()
    except:
        return "unknown"

def dir_size(path):
    total = 0
    if os.path.isdir(path):
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total += os.path.getsize(fp)
                except:
                    pass
    return f"{total / (1024**3):.1f} GB"

manifest = {
    "snapshot_date": datetime.datetime.utcnow().isoformat() + "Z",
    "hostname": run("hostname"),
    "gpu": run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"),
    "cuda_version": run("nvcc --version | tail -1"),
    "python_version": run("python3 --version"),
    "components": {},
}

# Check each component
components = {
    "sage_repo": "/workspace/SAGE",
    "sam3d_checkpoints": "/workspace/sam3d/checkpoints/hf",
    "sam3d_repo": "/workspace/sam3d/repo",
    "miniconda": "/workspace/miniconda3",
    "sage_conda_env": "/workspace/miniconda3/envs/sage",
    "isaacsim_env": "/workspace/isaacsim_env",
    "blueprint_scripts": "/workspace/BlueprintPipeline/scripts/runpod_sage",
    "entrypoint": "/workspace/entrypoint.sh",
    "patches": "/workspace/apply_sage_patches.sh",
    "physx_anything": "/workspace/PhysX-Anything",
    "physx_anything_weights": "/workspace/PhysX-Anything/pretrain/vlm",
    "infinigen": "/workspace/infinigen",
}

for name, path in components.items():
    exists = os.path.exists(path)
    manifest["components"][name] = {
        "path": path,
        "exists": exists,
        "size": dir_size(path) if exists and os.path.isdir(path) else ("file" if exists else "missing"),
    }

# Conda env packages
try:
    pkgs = run("conda run -n sage pip list --format=freeze 2>/dev/null | head -50")
    manifest["sage_env_packages_sample"] = pkgs.split('\n')[:20]
except:
    pass

# Check key files exist
key_files = [
    "/workspace/SAGE/client/key.json",
    "/workspace/SAGE/server/key.json",
    "/workspace/sam3d/checkpoints/hf/pipeline.yaml",
    "/workspace/isaacsim_env/bin/python3",
]
manifest["key_files"] = {f: os.path.exists(f) for f in key_files}

# IsaacSim path
isaacsim_path_file = "/workspace/.isaacsim_path"
if os.path.exists(isaacsim_path_file):
    with open(isaacsim_path_file) as f:
        manifest["isaacsim_path"] = f.read().strip()

print(json.dumps(manifest, indent=2))
PYEOF

log "Manifest written to ${MANIFEST}"
cat "${MANIFEST}"

# ── 2. Check for missing components ─────────────────────────────────────────
log ""
log "Checking components..."
MISSING=0

check_component() {
    local path="$1" name="$2"
    if [[ -e "${path}" ]]; then
        local size=$(du -sh "${path}" 2>/dev/null | cut -f1)
        log "  OK: ${name} (${size})"
    else
        log "  MISSING: ${name} at ${path}"
        MISSING=$((MISSING + 1))
    fi
}

check_component "/workspace/SAGE" "SAGE repo"
check_component "/workspace/sam3d/checkpoints/hf" "SAM3D checkpoints"
check_component "/workspace/miniconda3/envs/sage" "Conda env: sage"
check_component "/workspace/isaacsim_env" "Isaac Sim venv"
check_component "/workspace/BlueprintPipeline/scripts/runpod_sage/sam3d_server.py" "SAM3D server"
check_component "/workspace/BlueprintPipeline/scripts/runpod_sage/entrypoint_sage_sam3d.sh" "Entrypoint script"
check_component "/workspace/PhysX-Anything/pretrain/vlm" "PhysX-Anything weights"
check_component "/workspace/infinigen/scripts/spawn_asset.py" "Infinigen"

# ── Special check: Isaac Sim ────────────────────────────────────────────────
ISAACSIM_IMPORTABLE=0
if [[ -d "/workspace/isaacsim_env" ]]; then
    if /workspace/isaacsim_env/bin/python3 -c "import isaacsim" 2>/dev/null; then
        log "  Isaac Sim: IMPORTABLE (ready)"
        ISAACSIM_IMPORTABLE=1
    else
        log "  Isaac Sim: venv exists but NOT importable"
        MISSING=$((MISSING + 1))
    fi
else
    log ""
    log "  *** Isaac Sim is NOT installed on this pod. ***"
    log ""
    log "  To include Isaac Sim in the snapshot, run first:"
    log "    bash /workspace/BlueprintPipeline/scripts/runpod_sage/install_isaacsim_on_pod.sh"
    log ""
    log "  This takes ~10-20 minutes but means instant startup on new pods."
    log "  The L40S has enough VRAM for both SAM3D + Isaac Sim (~20-22GB of 48GB)."
    log ""
    MISSING=$((MISSING + 1))
fi

if [[ ${MISSING} -gt 0 ]]; then
    log ""
    log "WARNING: ${MISSING} component(s) missing. Snapshot will be incomplete."
    if [[ ${ISAACSIM_IMPORTABLE} -eq 0 ]] && [[ -d "/workspace/isaacsim_env" ]] || [[ ! -d "/workspace/isaacsim_env" ]]; then
        log "  Strongly recommend installing Isaac Sim first for a complete snapshot."
    fi
    read -rp "Continue anyway? (y/N) " choice
    if [[ ! "${choice}" =~ ^[Yy] ]]; then
        log "Aborted."
        exit 1
    fi
fi

# ── 3. Copy entrypoint + patch scripts if not already in place ───────────────
log ""
log "Ensuring entrypoint and patch scripts are in place..."

# Make sure the entrypoint is at /workspace/entrypoint.sh
if [[ -f "/workspace/BlueprintPipeline/scripts/runpod_sage/entrypoint_sage_sam3d.sh" ]]; then
    cp -f "/workspace/BlueprintPipeline/scripts/runpod_sage/entrypoint_sage_sam3d.sh" "/workspace/entrypoint.sh"
    chmod +x "/workspace/entrypoint.sh"
    log "  Copied entrypoint to /workspace/entrypoint.sh"
fi

if [[ -f "/workspace/BlueprintPipeline/scripts/runpod_sage/apply_sage_patches.sh" ]]; then
    cp -f "/workspace/BlueprintPipeline/scripts/runpod_sage/apply_sage_patches.sh" "/workspace/apply_sage_patches.sh"
    chmod +x "/workspace/apply_sage_patches.sh"
    log "  Copied patches to /workspace/apply_sage_patches.sh"
fi

# ── 4. Estimate archive size ────────────────────────────────────────────────
log ""
log "Estimating archive size..."
TOTAL_BYTES=0
for d in /workspace/SAGE /workspace/sam3d /workspace/miniconda3 /workspace/isaacsim_env /workspace/PhysX-Anything /workspace/infinigen; do
    if [[ -d "$d" ]]; then
        SIZE=$(du -sb "$d" 2>/dev/null | cut -f1)
        TOTAL_BYTES=$((TOTAL_BYTES + SIZE))
    fi
done
TOTAL_GB=$(echo "scale=1; ${TOTAL_BYTES} / 1073741824" | bc 2>/dev/null || echo "?")
COMPRESSED_GB=$(echo "scale=1; ${TOTAL_BYTES} * 0.6 / 1073741824" | bc 2>/dev/null || echo "?")
log "  Uncompressed: ~${TOTAL_GB} GB"
log "  Estimated compressed: ~${COMPRESSED_GB} GB"
log ""

# Check available disk space
AVAIL=$(df -BG /workspace 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G')
log "  Available disk: ${AVAIL}G"
if [[ -n "${AVAIL}" ]] && [[ "${AVAIL}" -lt 30 ]]; then
    log "WARNING: Less than 30GB free. Archive creation may fail."
    log "  Consider cleaning up first: pip cache purge, conda clean -afy"
fi

# ── 5. Clean up before archiving ────────────────────────────────────────────
log "Cleaning up caches to reduce size..."
pip cache purge 2>/dev/null || true
conda clean -afy 2>/dev/null || true
find /workspace -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
find /workspace -name '*.pyc' -delete 2>/dev/null || true
rm -rf /tmp/pip-* /tmp/conda-* 2>/dev/null || true

# Remove key.json files (contain API keys — will be regenerated by entrypoint)
log "Stripping API keys from key.json files (regenerated at startup)..."
for kf in /workspace/SAGE/client/key.json /workspace/SAGE/server/key.json; do
    if [[ -f "${kf}" ]]; then
        python3 -c "
import json
with open('${kf}') as f: d = json.load(f)
for k in d:
    if 'KEY' in k or 'TOKEN' in k:
        d[k] = 'REPLACE_AT_RUNTIME'
    if isinstance(d[k], dict):
        for kk in d[k]:
            if 'KEY' in kk or 'TOKEN' in kk:
                d[k][kk] = 'REPLACE_AT_RUNTIME'
with open('${kf}', 'w') as f: json.dump(d, f, indent=4)
" 2>/dev/null || true
    fi
done

# Remove runtime secret file (rehydrated on container start)
if [[ -f /workspace/.sage_runpod_secrets.env ]]; then
    log "Removing runtime secret file from snapshot: /workspace/.sage_runpod_secrets.env"
    rm -f /workspace/.sage_runpod_secrets.env
fi

# ── 6. Create archive ───────────────────────────────────────────────────────
log ""
log "Creating archive: ${ARCHIVE}"
log "This will take 10-30 minutes depending on disk speed..."
log ""

tar czf "${ARCHIVE}" \
    --exclude='*.log' \
    --exclude='*.tmp' \
    --exclude='.cache/pip' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='workspace/BlueprintPipeline/.git' \
    --exclude='workspace/BlueprintPipeline/.venv' \
    --exclude='workspace/BlueprintPipeline/.venv-text-backends' \
    --exclude='workspace/BlueprintPipeline/analysis_outputs' \
    --exclude='workspace/BlueprintPipeline/htmlcov' \
    --exclude='workspace/BlueprintPipeline/local_runs' \
    --exclude='workspace/BlueprintPipeline/downloaded_episodes' \
    --exclude='workspace/BlueprintPipeline/dead_letter' \
    --exclude='workspace/.sage_runpod_secrets.env' \
    --exclude='workspace/PhysX-Anything/.git' \
    --exclude='workspace/infinigen/.git' \
    --warning=no-file-changed \
    -C / \
    workspace/SAGE \
    workspace/sam3d \
    workspace/miniconda3 \
    workspace/isaacsim_env \
    workspace/BlueprintPipeline \
    workspace/entrypoint.sh \
    workspace/apply_sage_patches.sh \
    workspace/.isaacsim_path \
    workspace/PhysX-Anything \
    workspace/infinigen \
    workspace/infinigen_service.py \
    workspace/physx_anything_service.py \
    "${MANIFEST}" \
    2>/dev/null || true

ARCHIVE_SIZE=$(du -sh "${ARCHIVE}" 2>/dev/null | cut -f1)
SHA=$(sha256sum "${ARCHIVE}" | cut -d' ' -f1)
echo "${SHA}  $(basename "${ARCHIVE}")" > "${ARCHIVE}.sha256"

log ""
log "=========================================="
log "Snapshot complete!"
log "=========================================="
log ""
log "Archive: ${ARCHIVE} (${ARCHIVE_SIZE})"
log "SHA256:  ${SHA}"
log ""
log "To download to your local machine:"
log "  scp -P <PORT> root@<IP>:${ARCHIVE} ."
log ""
log "To import on a new pod/machine:"
log "  tar xzf $(basename ${ARCHIVE}) -C /"
log "  bash /workspace/entrypoint.sh"
log ""
log "To build a Docker image from this archive:"
log "  1. Start a base container:"
log "     docker run -d --name sage-build runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 sleep infinity"
log "  2. Copy archive in:"
log "     docker cp $(basename ${ARCHIVE}) sage-build:/tmp/"
log "  3. Extract inside:"
log "     docker exec sage-build tar xzf /tmp/$(basename ${ARCHIVE}) -C /"
log "  4. Install system deps:"
log "     docker exec sage-build bash -c 'apt-get update && apt-get install -y libgl1 libglib2.0-0 libxrender1 libxi6 libxxf86vm1 libxfixes3 libxkbcommon0 libsm6 libice6 libxext6 libxrandr2 libxcursor1 libxinerama1 libepoxy0 libglu1-mesa libegl1 libopengl0 libglx-mesa0 psmisc iproute2'"
log "  5. Commit:"
log "     docker commit --change 'CMD [\"/workspace/entrypoint.sh\"]' --change 'EXPOSE 8080' sage-build sage-sam3d:latest"
log "  6. Push:"
log "     docker tag sage-sam3d:latest yourusername/sage-sam3d:latest"
log "     docker push yourusername/sage-sam3d:latest"
