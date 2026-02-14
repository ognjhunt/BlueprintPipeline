#!/usr/bin/env bash
# =============================================================================
# Instant-start entrypoint for SAGE + SAM3D Docker image.
#
# This script is the CMD of the Docker image. It:
#   1. Writes key.json files from env vars (API keys)
#   2. Applies SAGE patches (idempotent)
#   3. Starts SAM3D server on :8080 (replaces TRELLIS)
#   4. Starts Isaac Sim headless with MCP extension
#   5. Waits for both services to be healthy
#   6. Keeps container alive (tail -f)
#
# Required env vars:
#   OPENAI_API_KEY  — OpenAI API key
#
# Recommended env vars:
#   GEMINI_API_KEY  — Gemini API key (primary image gen backend)
#   SLURM_JOB_ID   — Used by SAGE MCP port hashing (default: 12345)
#
# Optional env vars:
#   OPENAI_MODEL         — Model name (default: gpt-5.1)
#   OPENAI_BASE_URL      — Base URL (default: https://api.openai.com/v1)
#   ANTHROPIC_API_KEY    — Claude API key
#   SAM3D_IMAGE_BACKEND  — gemini or openai (default: gemini)
#   SAM3D_PORT           — SAM3D server port (default: 8080)
#   SKIP_ISAAC_SIM       — Set to 1 to skip Isaac Sim startup
#   SKIP_SAM3D           — Set to 1 to skip SAM3D startup
#   SAGE_ONLY            — Set to 1 to only write configs (manual start)
# =============================================================================
set -euo pipefail

log() { echo "[sage-entrypoint $(date -u +%FT%TZ)] $*"; }

WORKSPACE=${WORKSPACE:-/workspace}
SAGE_DIR="${WORKSPACE}/SAGE"
SAM3D_PORT=${SAM3D_PORT:-8080}
SLURM_JOB_ID=${SLURM_JOB_ID:-12345}
OPENAI_MODEL="${OPENAI_MODEL:-gpt-5.1}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
SAM3D_IMAGE_BACKEND="${SAM3D_IMAGE_BACKEND:-gemini}"
SAM3D_TEXTURE_BAKING="${SAM3D_TEXTURE_BAKING:-1}"

export SLURM_JOB_ID
export SAM3D_TEXTURE_BAKING

log "=========================================="
log "SAGE + SAM3D — Instant Start Entrypoint"
log "=========================================="

# ── 0. GPU Check ─────────────────────────────────────────────────────────────
if command -v nvidia-smi >/dev/null 2>&1; then
    log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
else
    log "WARNING: No GPU detected. SAM3D and Isaac Sim require a GPU."
fi

# ── 1. Write key.json files from env vars ────────────────────────────────────
log "Writing SAGE key.json files..."

if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    # Client key.json
    python3 - <<PYEOF
import json, os, pathlib

client_key = {
    "API_TOKEN": os.environ.get("OPENAI_API_KEY", ""),
    "API_URL_QWEN": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    "MODEL_NAME": os.environ.get("OPENAI_MODEL", "gpt-5.1"),
}
path = pathlib.Path("${SAGE_DIR}/client/key.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(client_key, indent=4) + "\n")
path.chmod(0o600)
print(f"  wrote {path}")

# Server key.json
model = os.environ.get("OPENAI_MODEL", "gpt-5.1")
server_key = {
    "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
    "API_TOKEN": os.environ.get("OPENAI_API_KEY", ""),
    "API_URL_QWEN": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    "API_URL_OPENAI": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    "MODEL_DICT": {
        "qwen": model,
        "openai": model,
        "glmv": model,
        "claude": model,
    },
    "TRELLIS_SERVER_URL": f"http://localhost:{os.environ.get('SAM3D_PORT', '8080')}",
    "FLUX_SERVER_URL": "",
    "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY", ""),
}
path = pathlib.Path("${SAGE_DIR}/server/key.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(server_key, indent=4) + "\n")
path.chmod(0o600)
print(f"  wrote {path}")
PYEOF
    log "key.json files written."
else
    log "WARNING: OPENAI_API_KEY not set. key.json not written."
    log "  Set it before running SAGE: export OPENAI_API_KEY=sk-..."
fi

# ── 1.5. Pull latest patches from GitHub (if repo available) ─────────────────
BP_DIR="${WORKSPACE}/BlueprintPipeline"
PATCH_REPO="${SAGE_PATCH_REPO:-https://github.com/ognjhunt/BlueprintPipeline.git}"
PATCH_BRANCH="${SAGE_PATCH_BRANCH:-main}"

if [[ "${SKIP_PATCH_PULL:-0}" != "1" ]]; then
    if [[ -d "${BP_DIR}/.git" ]]; then
        log "Pulling latest patches from GitHub..."
        (cd "${BP_DIR}" && git pull --ff-only origin "${PATCH_BRANCH}" 2>&1 | head -5) || \
            log "  WARNING: git pull failed (using baked-in patches)"
        # Update the entrypoint-adjacent scripts
        if [[ -f "${BP_DIR}/scripts/runpod_sage/apply_sage_patches.sh" ]]; then
            cp -f "${BP_DIR}/scripts/runpod_sage/apply_sage_patches.sh" "${WORKSPACE}/apply_sage_patches.sh"
            chmod +x "${WORKSPACE}/apply_sage_patches.sh"
        fi
        log "  Patches updated from GitHub (branch: ${PATCH_BRANCH})"
    elif command -v git >/dev/null 2>&1; then
        log "No BlueprintPipeline repo found. Cloning patch scripts..."
        git clone --depth 1 --branch "${PATCH_BRANCH}" "${PATCH_REPO}" "${BP_DIR}" 2>&1 | tail -3 || \
            log "  WARNING: git clone failed (using baked-in patches only)"
        if [[ -f "${BP_DIR}/scripts/runpod_sage/apply_sage_patches.sh" ]]; then
            cp -f "${BP_DIR}/scripts/runpod_sage/apply_sage_patches.sh" "${WORKSPACE}/apply_sage_patches.sh"
            chmod +x "${WORKSPACE}/apply_sage_patches.sh"
        fi
    else
        log "  git not available — using baked-in patches only"
    fi
else
    log "Skipping patch pull (SKIP_PATCH_PULL=1)"
fi

# ── 2. Apply SAGE patches (idempotent) ──────────────────────────────────────
if [[ -x "${WORKSPACE}/apply_sage_patches.sh" ]]; then
    log "Applying SAGE patches..."
    bash "${WORKSPACE}/apply_sage_patches.sh" || log "WARNING: Some patches may have failed (non-fatal)"
else
    log "No patch script found (patches may already be baked into image)."
fi

# ── 3. Early exit if SAGE_ONLY mode ─────────────────────────────────────────
if [[ "${SAGE_ONLY:-0}" == "1" ]]; then
    log "SAGE_ONLY=1 — configs written. Keeping container alive for manual operation."
    log "To run SAGE:"
    log "  source /workspace/miniconda3/etc/profile.d/conda.sh && conda activate sage"
    log "  cd /workspace/SAGE/client"
    log "  python client_generation_room_desc.py --room_desc '...' --server_paths ../server/layout_wo_robot.py"
    exec tail -f /dev/null
fi

# ── 4. Start SAM3D server ───────────────────────────────────────────────────
if [[ "${SKIP_SAM3D:-0}" != "1" ]]; then
    SAM3D_SCRIPT="${WORKSPACE}/BlueprintPipeline/scripts/runpod_sage/sam3d_server.py"
    if [[ ! -f "${SAM3D_SCRIPT}" ]]; then
        log "ERROR: sam3d_server.py not found at ${SAM3D_SCRIPT}"
    else
        log "Starting SAM3D server on :${SAM3D_PORT} (backend=${SAM3D_IMAGE_BACKEND})..."

        SAM3D_ARGS="--port ${SAM3D_PORT} --image-backend ${SAM3D_IMAGE_BACKEND}"
        [[ -n "${OPENAI_API_KEY:-}" ]] && SAM3D_ARGS="${SAM3D_ARGS} --openai-key ${OPENAI_API_KEY}"
        [[ -n "${GEMINI_API_KEY:-}" ]] && SAM3D_ARGS="${SAM3D_ARGS} --gemini-key ${GEMINI_API_KEY}"

        # Check for SAM3D checkpoints
        if [[ -f "/workspace/sam3d/checkpoints/hf/pipeline.yaml" ]]; then
            SAM3D_ARGS="${SAM3D_ARGS} --checkpoint-dir /workspace/sam3d/checkpoints/hf"
        else
            log "WARNING: SAM3D checkpoints not found at /workspace/sam3d/checkpoints/hf"
            log "  Download with: huggingface-cli download facebook/sam-3d-objects --local-dir /workspace/sam3d/checkpoints/hf"
        fi

        nohup python3 "${SAM3D_SCRIPT}" ${SAM3D_ARGS} \
            > /tmp/sam3d_server.log 2>&1 &
        SAM3D_PID=$!
        log "SAM3D PID: ${SAM3D_PID}"

        # Wait for SAM3D health (model loading can take 30-90s)
        log "Waiting for SAM3D health on :${SAM3D_PORT}..."
        deadline=$(( $(date +%s) + 300 ))
        while [[ "$(date +%s)" -lt "${deadline}" ]]; do
            if curl -sf "http://127.0.0.1:${SAM3D_PORT}/health" >/dev/null 2>&1; then
                log "SAM3D server healthy!"
                break
            fi
            if ! kill -0 "${SAM3D_PID}" 2>/dev/null; then
                log "ERROR: SAM3D server died. Last 40 lines:"
                tail -40 /tmp/sam3d_server.log 2>/dev/null || true
                break
            fi
            sleep 3
        done
    fi
else
    log "Skipping SAM3D startup (SKIP_SAM3D=1)"
fi

# ── 5. Start Isaac Sim ──────────────────────────────────────────────────────
if [[ "${SKIP_ISAAC_SIM:-0}" != "1" ]]; then
    log "Starting Isaac Sim headless with MCP extension..."
    export OMNI_KIT_ACCEPT_EULA=YES
    export ACCEPT_EULA=Y
    export PRIVACY_CONSENT=Y

    # Get Isaac Sim path
    ISAACSIM_PATH=""
    KIT_FILE=""
    LAUNCH_METHOD=""
    if [[ -f /workspace/.isaacsim_path ]]; then
        source /workspace/.isaacsim_path
    fi
    if [[ -z "${ISAACSIM_PATH:-}" ]]; then
        # Try to find it from the venv
        ISAACSIM_PATH=$(find /workspace/isaacsim_env -path "*/isaacsim/__init__.py" -exec dirname {} \; 2>/dev/null | head -1)
    fi

    if [[ -z "${ISAACSIM_PATH}" ]] && [[ ! -d "/workspace/isaacsim_env" ]]; then
        log "ERROR: Cannot find Isaac Sim installation."
    else
        # Find kit file if not set
        if [[ -z "${KIT_FILE}" ]]; then
            for candidate in \
                "${ISAACSIM_PATH}/apps/isaacsim.exp.full.kit" \
                "${ISAACSIM_PATH}/apps/isaacsim.exp.base.kit" \
                "${ISAACSIM_PATH}/apps/omni.isaac.sim.kit"; do
                if [[ -f "${candidate}" ]]; then
                    KIT_FILE="${candidate}"
                    break
                fi
            done
        fi

        if [[ -z "${KIT_FILE}" ]]; then
            KIT_FILE=$(find "${ISAACSIM_PATH}/apps" -name "*.kit" -print -quit 2>/dev/null || true)
        fi

        if [[ -z "${KIT_FILE}" ]]; then
            log "ERROR: No .kit experience file found."
        else
            log "  Isaac Sim path: ${ISAACSIM_PATH}"
            log "  Kit file: ${KIT_FILE}"
            log "  SLURM_JOB_ID: ${SLURM_JOB_ID}"

            # Determine launch method: pip install uses 'python -m isaacsim', Docker uses kit/kit binary
            if [[ -x "${ISAACSIM_PATH}/kit/kit" ]]; then
                log "  Launch: kit/kit binary"
                nohup "${ISAACSIM_PATH}/kit/kit" \
                    "${KIT_FILE}" \
                    --no-window \
                    --enable isaac.sim.mcp_extension \
                    > /tmp/isaacsim.log 2>&1 &
            else
                log "  Launch: python -m isaacsim (pip install)"
                nohup /workspace/isaacsim_env/bin/python3 -m isaacsim \
                    "${KIT_FILE}" \
                    --no-window \
                    --enable isaac.sim.mcp_extension \
                    > /tmp/isaacsim.log 2>&1 &
            fi
            ISAAC_PID=$!
            log "Isaac Sim PID: ${ISAAC_PID}"

            # Compute expected MCP port
            MCP_PORT=$(python3 -c "
import hashlib, os
job_id = os.environ.get('SLURM_JOB_ID', '12345')
h = int(hashlib.md5(str(job_id).encode()).hexdigest(), 16)
print(8080 + (h % (40000 - 8080 + 1)))
")
            log "Expected MCP port: ${MCP_PORT}"

            # Wait for MCP port
            log "Waiting for Isaac Sim MCP on :${MCP_PORT}..."
            deadline=$(( $(date +%s) + 600 ))
            while [[ "$(date +%s)" -lt "${deadline}" ]]; do
                if ss -lnt 2>/dev/null | awk '{print $4}' | grep -q ":${MCP_PORT}$"; then
                    log "Isaac Sim MCP ready on :${MCP_PORT}!"
                    break
                fi
                if ! kill -0 "${ISAAC_PID}" 2>/dev/null; then
                    log "ERROR: Isaac Sim died. Last 60 lines:"
                    tail -60 /tmp/isaacsim.log 2>/dev/null || true
                    break
                fi
                sleep 5
            done
        fi
    fi
else
    log "Skipping Isaac Sim startup (SKIP_ISAAC_SIM=1)"
fi

# ── 6. Final status ─────────────────────────────────────────────────────────
log "=========================================="
log "Startup complete."
log ""
log "Services:"
SAM3D_HEALTH=$(curl -sf "http://127.0.0.1:${SAM3D_PORT}/health" 2>/dev/null || echo "not running")
log "  SAM3D:    http://0.0.0.0:${SAM3D_PORT} (${SAM3D_HEALTH})"
if [[ "${SKIP_ISAAC_SIM:-0}" != "1" ]]; then
    log "  Isaac Sim: MCP port ${MCP_PORT:-unknown}"
fi
log ""
log "To run SAGE:"
log "  source /workspace/miniconda3/etc/profile.d/conda.sh"
log "  conda activate sage"
log "  cd /workspace/SAGE/client"
log "  python client_generation_room_desc.py \\"
log "    --room_desc 'A modern living room with a sofa and coffee table' \\"
log "    --server_paths ../server/layout_wo_robot.py"
log ""
log "Logs:"
log "  SAM3D:     tail -f /tmp/sam3d_server.log"
log "  Isaac Sim: tail -f /tmp/isaacsim.log"
log "=========================================="

# Keep container alive
exec tail -f /dev/null
