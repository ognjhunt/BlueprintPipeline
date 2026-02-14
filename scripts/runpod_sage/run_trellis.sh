#!/usr/bin/env bash
# run_trellis.sh — Lightweight TRELLIS server start (NO reinstall).
# Use after bootstrap_trellis_pod.sh has been run once, or after
# importing a snapshot via import_trellis_snapshot.sh.
#
# This script:
#   1. Verifies conda env + key packages exist (fails fast if not)
#   2. Activates the trellis conda env
#   3. Starts the central+worker server
#   4. Pre-warms the model with a dummy generation request
#   5. Reports healthy
#
# Typical cold-start time: 60-90s (model load) vs 15-20 min (full bootstrap)
set -euo pipefail

log() { echo "[run-trellis $(date -u +%FT%TZ)] $*"; }

WORKSPACE=${WORKSPACE:-/workspace}
TRELLIS_DIR="${WORKSPACE}/SAGE/server/TRELLIS"
SERVER_PY="${WORKSPACE}/SAGE/server/TRELLIS/server.py"
CENTRAL_PY="${WORKSPACE}/SAGE/server/TRELLIS/central_server.py"
LOG_PATH="/tmp/trellis_worker.log"
CENTRAL_LOG="/tmp/trellis_central.log"
PORT=${TRELLIS_PORT:-8080}
WORKER_PORT=${TRELLIS_WORKER_PORT:-8081}

# ──────────────────────────────────────────────
# 0. Quick check: is it already running?
# ──────────────────────────────────────────────
if curl -sf "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q '"healthy"'; then
    log "TRELLIS already healthy on :${PORT}. Nothing to do."
    exit 0
fi

# ──────────────────────────────────────────────
# 1. Preflight checks (fail fast, not fail slow)
# ──────────────────────────────────────────────
log "Preflight checks..."

if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "ERROR: nvidia-smi not found — no GPU."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

CONDA_BASE="/root/miniconda3"
if [[ ! -d "${CONDA_BASE}/envs/trellis" ]]; then
    log "ERROR: Conda env 'trellis' not found. Run bootstrap_trellis_pod.sh first."
    exit 2
fi

if [[ ! -f "${SERVER_PY}" ]]; then
    log "ERROR: server.py not found at ${SERVER_PY}."
    log "  Run bootstrap_trellis_pod.sh first, or import a snapshot."
    exit 2
fi

# Activate conda
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate trellis

# Verify key packages can import
python -c "import torch; assert torch.cuda.is_available(), 'No CUDA'" 2>/dev/null || {
    log "ERROR: torch CUDA not available in trellis env."
    exit 3
}
python -c "import nvdiffrast" 2>/dev/null || {
    log "WARNING: nvdiffrast missing — attempting install with --no-build-isolation"
    pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation 2>/dev/null || {
        log "ERROR: Failed to install nvdiffrast."
        exit 3
    }
}
python -c "from trellis.pipelines import TrellisTextTo3DPipeline" 2>/dev/null || {
    log "ERROR: TRELLIS pipeline not importable. Environment may be corrupt."
    exit 3
}

log "Preflight OK."

# ──────────────────────────────────────────────
# 2. Kill any leftover processes
# ──────────────────────────────────────────────
log "Cleaning up stale processes..."
pkill -f "python server.py --port ${WORKER_PORT}" 2>/dev/null || true
pkill -f "python central_server.py" 2>/dev/null || true
sleep 1

# ──────────────────────────────────────────────
# 3. Set environment
# ──────────────────────────────────────────────
export LD_PRELOAD=${LD_PRELOAD:-/usr/lib/x86_64-linux-gnu/libstdc++.so.6}
export ATTN_BACKEND=xformers
export SPCONV_ALGO=native

# ──────────────────────────────────────────────
# 4. Start worker server (loads model into GPU)
# ──────────────────────────────────────────────
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
log "Starting worker on GPU 0 (of ${GPU_COUNT} available), port ${WORKER_PORT}"

cd "${TRELLIS_DIR}"
CUDA_VISIBLE_DEVICES=0 nohup python server.py --port "${WORKER_PORT}" --gpu 0 \
    > "${LOG_PATH}" 2>&1 &
WORKER_PID=$!
log "Worker PID: ${WORKER_PID}"

# Wait for worker to become healthy (model loading = 60-120s)
log "Waiting for worker on :${WORKER_PORT} (model loading ~60-120s)..."
deadline=$(( $(date +%s) + 300 ))  # 5 min timeout
while [[ "$(date +%s)" -lt "${deadline}" ]]; do
    if curl -sf "http://127.0.0.1:${WORKER_PORT}/health" >/dev/null 2>&1; then
        log "Worker healthy on :${WORKER_PORT}"
        break
    fi
    # Check worker hasn't crashed
    if ! kill -0 "${WORKER_PID}" 2>/dev/null; then
        log "ERROR: Worker process died. Last 30 lines:"
        tail -30 "${LOG_PATH}" 2>/dev/null || true
        exit 4
    fi
    sleep 5
done

if ! curl -sf "http://127.0.0.1:${WORKER_PORT}/health" >/dev/null 2>&1; then
    log "ERROR: Worker did not become healthy within 5 minutes."
    tail -30 "${LOG_PATH}" 2>/dev/null || true
    exit 4
fi

# ──────────────────────────────────────────────
# 5. Start central distributor
# ──────────────────────────────────────────────
log "Starting central distributor on :${PORT}"
nohup python central_server.py "http://localhost:${WORKER_PORT}" \
    > "${CENTRAL_LOG}" 2>&1 &
CENTRAL_PID=$!
sleep 3

if ! curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    log "ERROR: Central server failed to start."
    tail -20 "${CENTRAL_LOG}" 2>/dev/null || true
    exit 5
fi
log "Central distributor healthy on :${PORT}"

# ──────────────────────────────────────────────
# 6. Pre-warm: send a dummy generation request
# ──────────────────────────────────────────────
if [[ "${SKIP_PREWARM:-0}" != "1" ]]; then
    log "Pre-warming model with dummy request (first gen compiles CUDA kernels)..."
    PREWARM_RESPONSE=$(curl -sf -X POST "http://127.0.0.1:${PORT}/generate" \
        -H "Content-Type: application/json" \
        -d '{"input_text": "a simple red cube"}' 2>/dev/null || echo '{}')

    JOB_ID=$(echo "${PREWARM_RESPONSE}" | python -c "import sys,json; print(json.load(sys.stdin).get('job_id',''))" 2>/dev/null || echo "")

    if [[ -n "${JOB_ID}" ]]; then
        log "Pre-warm job submitted: ${JOB_ID}. Waiting for completion..."
        prewarm_deadline=$(( $(date +%s) + 300 ))
        while [[ "$(date +%s)" -lt "${prewarm_deadline}" ]]; do
            STATUS=$(curl -sf "http://127.0.0.1:${PORT}/job/${JOB_ID}" -o /dev/null -w "%{http_code}" 2>/dev/null || echo "000")
            if [[ "${STATUS}" == "200" ]]; then
                log "Pre-warm complete. Model is hot."
                break
            elif [[ "${STATUS}" == "500" ]]; then
                log "WARNING: Pre-warm generation failed (non-fatal). Model loaded but first real request may be slow."
                break
            fi
            sleep 5
        done
    else
        log "WARNING: Could not submit pre-warm request (non-fatal)."
    fi
else
    log "Skipping pre-warm (SKIP_PREWARM=1)"
fi

# ──────────────────────────────────────────────
# 7. Final health report
# ──────────────────────────────────────────────
HEALTH=$(curl -sf "http://127.0.0.1:${PORT}/health" 2>/dev/null || echo '{"status":"unknown"}')
log "TRELLIS server ready."
log "  Health: ${HEALTH}"
log "  Central: http://0.0.0.0:${PORT} (PID ${CENTRAL_PID})"
log "  Worker:  http://0.0.0.0:${WORKER_PORT} (PID ${WORKER_PID})"
log "  Logs:    ${LOG_PATH} (worker), ${CENTRAL_LOG} (central)"
