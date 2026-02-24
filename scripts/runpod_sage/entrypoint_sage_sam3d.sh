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
#   OPENAI_API_KEY or OPENROUTER_API_KEY — OpenAI-compatible API key
#
# Recommended env vars:
#   GEMINI_API_KEY  — Gemini API key (primary image gen backend)
#   SLURM_JOB_ID   — Used by SAGE MCP port hashing (default: 12345)
#
# Optional env vars:
#   OPENAI_MODEL         — Legacy fallback model name (default: gpt-5.1)
#   OPENAI_BASE_URL      — Base URL (default: OPENROUTER_BASE_URL or https://api.openai.com/v1)
#   OPENAI_WEBSOCKET_BASE_URL — OpenAI Responses websocket endpoint (e.g. wss://api.openai.com/ws/v1/realtime?provider=openai)
#   OPENAI_USE_WEBSOCKET — Enable websocket mode for OpenAI responses (1|true|on, default: 1)
#   OPENROUTER_BASE_URL  — OpenRouter base URL (default: https://openrouter.ai/api/v1)
#   OPENAI_MODEL_QWEN    — qwen model (default: qwen/qwen3.5-397b-a17b)
#   OPENAI_MODEL_OPENAI  — openai slot model (default: moonshotai/kimi-k2.5)
#   OPENAI_MODEL_GLMV    — glmv slot model (default: OPENAI_MODEL)
#   OPENAI_MODEL_CLAUDE  — claude slot model (default: OPENAI_MODEL)
#   ANTHROPIC_API_KEY    — Claude API key
#   SAM3D_IMAGE_BACKEND  — gemini or openai (default: gemini)
#   SAM3D_GEMINI_IMAGE_MODELS — comma-separated Gemini image models (default: gemini-2.5-flash-image)
#   SAM3D_ENABLE_OPENAI_FALLBACK — set 1/true to allow fallback when Gemini fails (default: false)
#   SAM3D_PORT           — SAM3D server port (default: 8080)
#   SKIP_ISAAC_SIM       — Set to 1 to skip Isaac Sim startup
#   SKIP_SAM3D           — Set to 1 to skip SAM3D startup
#   SAGE_ONLY            — Set to 1 to only write configs (manual start)
# =============================================================================
set -euo pipefail

log() { echo "[sage-entrypoint $(date -u +%FT%TZ)] $*"; }

WORKSPACE=${WORKSPACE:-/workspace}
SAGE_DIR="${WORKSPACE}/SAGE"
BP_DIR="${WORKSPACE}/BlueprintPipeline"
MCP_RESOLVER="${BP_DIR}/scripts/runpod_sage/mcp_extension_paths.py"
SAM3D_PORT=${SAM3D_PORT:-8080}
SLURM_JOB_ID=${SLURM_JOB_ID:-12345}
OPENAI_MODEL="${OPENAI_MODEL:-gpt-5.1}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-${OPENROUTER_BASE_URL:-https://api.openai.com/v1}}"
OPENAI_WEBSOCKET_BASE_URL="${OPENAI_WEBSOCKET_BASE_URL:-}"
if [[ -z "${OPENAI_WEBSOCKET_BASE_URL}" && "${OPENAI_BASE_URL,,}" == *"api.openai.com"* ]]; then
  OPENAI_WEBSOCKET_BASE_URL="wss://api.openai.com/ws/v1/realtime?provider=openai"
fi
OPENAI_USE_WEBSOCKET="${OPENAI_USE_WEBSOCKET:-1}"
OPENAI_MODEL_QWEN="${OPENAI_MODEL_QWEN:-qwen/qwen3.5-397b-a17b}"
OPENAI_MODEL_OPENAI="${OPENAI_MODEL_OPENAI:-moonshotai/kimi-k2.5}"
OPENAI_MODEL_GLMV="${OPENAI_MODEL_GLMV:-${OPENAI_MODEL}}"
OPENAI_MODEL_CLAUDE="${OPENAI_MODEL_CLAUDE:-${OPENAI_MODEL}}"
SAM3D_IMAGE_BACKEND="${SAM3D_IMAGE_BACKEND:-gemini}"
SAM3D_GEMINI_IMAGE_MODELS="${SAM3D_GEMINI_IMAGE_MODELS:-gemini-2.5-flash-image}"
SAM3D_ENABLE_OPENAI_FALLBACK="${SAM3D_ENABLE_OPENAI_FALLBACK:-0}"
SAM3D_TEXTURE_BAKING="${SAM3D_TEXTURE_BAKING:-1}"
SAM3D_AUTO_INSTALL_DEPS="${SAM3D_AUTO_INSTALL_DEPS:-1}"
SKIP_PATCHES="${SKIP_PATCHES:-0}"
RUN_MODE="${RUN_MODE:-services}"  # services|full_pipeline

is_placeholder() {
    # Treat common placeholder tokens as "unset" so we don't bake them into key.json.
    # Usage: is_placeholder "${VAR:-}"
    local v="${1:-}"
    v="$(echo "${v}" | tr -d '[:space:]')"
    if [[ -z "${v}" ]]; then
        return 0
    fi
    v="${v^^}"
    [[ "${v}" == "PLACEHOLDER" || "${v}" == "CHANGEME" || "${v}" == "TODO" ]]
}

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

# Full pipeline launches its own headless SimulationApp in Stage 7; the MCP service
# only wastes VRAM and can destabilize stdio. Default to skipping Isaac Sim here.
if [[ "${RUN_MODE}" == "full_pipeline" ]] && [[ -z "${SKIP_ISAAC_SIM:-}" ]]; then
    SKIP_ISAAC_SIM=1
fi

export SLURM_JOB_ID
export SAM3D_TEXTURE_BAKING
export SAM3D_PORT
export SAM3D_GEMINI_IMAGE_MODELS
export SAM3D_ENABLE_OPENAI_FALLBACK
export SAM3D_AUTO_INSTALL_DEPS
export OPENAI_WEBSOCKET_BASE_URL
export OPENAI_USE_WEBSOCKET
export REQUIRE_LOCAL_ROBOT_ASSET="${REQUIRE_LOCAL_ROBOT_ASSET:-1}"
export SAGE_ALLOW_REMOTE_ISAAC_ASSETS="${SAGE_ALLOW_REMOTE_ISAAC_ASSETS:-0}"
export SAGE_SENSOR_FAILURE_POLICY="${SAGE_SENSOR_FAILURE_POLICY:-fail}"
export SAGE_RENDER_WARMUP_FRAMES="${SAGE_RENDER_WARMUP_FRAMES:-100}"
export SAGE_SENSOR_MIN_RGB_STD="${SAGE_SENSOR_MIN_RGB_STD:-0.01}"

log "=========================================="
log "SAGE + SAM3D — Instant Start Entrypoint"
log "=========================================="

# Load runtime secrets from mounted env file (if present).
if [[ -f /workspace/.sage_runpod_secrets.env ]]; then
    # shellcheck disable=SC1091
    set -a
    source /workspace/.sage_runpod_secrets.env
    set +a
fi

# Keep SAM3D texture import path stable.
export LIDRA_SKIP_INIT=1

# ── 0. GPU Check ─────────────────────────────────────────────────────────────
if command -v nvidia-smi >/dev/null 2>&1; then
    log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
else
    log "WARNING: No GPU detected. SAM3D and Isaac Sim require a GPU."
fi

# ── 1. Write key.json files from env vars ────────────────────────────────────
log "Writing SAGE key.json files..."

OPENAI_API_KEY_EFFECTIVE="${OPENAI_API_KEY:-${OPENROUTER_API_KEY:-${API_TOKEN:-}}}"
GEMINI_API_KEY_EFFECTIVE="${GEMINI_API_KEY:-}"
ANTHROPIC_API_KEY_EFFECTIVE="${ANTHROPIC_API_KEY:-}"

if is_placeholder "${OPENAI_API_KEY_EFFECTIVE}"; then OPENAI_API_KEY_EFFECTIVE=""; fi
if is_placeholder "${GEMINI_API_KEY_EFFECTIVE}"; then GEMINI_API_KEY_EFFECTIVE=""; fi
if is_placeholder "${ANTHROPIC_API_KEY_EFFECTIVE}"; then ANTHROPIC_API_KEY_EFFECTIVE=""; fi

export OPENAI_API_KEY_EFFECTIVE GEMINI_API_KEY_EFFECTIVE ANTHROPIC_API_KEY_EFFECTIVE

if [[ -n "${OPENAI_API_KEY_EFFECTIVE}" ]]; then
    # Client key.json
    python3 - <<PYEOF
import json, os, pathlib

client_key = {
    "API_TOKEN": os.environ.get("OPENAI_API_KEY_EFFECTIVE", ""),
    "API_URL_QWEN": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    "MODEL_NAME": os.environ.get("OPENAI_MODEL_QWEN", "qwen/qwen3.5-397b-a17b"),
}
path = pathlib.Path("${SAGE_DIR}/client/key.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(client_key, indent=4) + "\n")
path.chmod(0o600)
print(f"  wrote {path}")

# Server key.json
model_qwen = os.environ.get("OPENAI_MODEL_QWEN", "qwen/qwen3.5-397b-a17b")
model_openai = os.environ.get("OPENAI_MODEL_OPENAI", "moonshotai/kimi-k2.5")
model_glmv = os.environ.get("OPENAI_MODEL_GLMV", os.environ.get("OPENAI_MODEL", "gpt-5.1"))
model_claude = os.environ.get("OPENAI_MODEL_CLAUDE", os.environ.get("OPENAI_MODEL", "gpt-5.1"))
server_key = {
    "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY_EFFECTIVE", ""),
    "API_TOKEN": os.environ.get("OPENAI_API_KEY_EFFECTIVE", ""),
    "API_URL_QWEN": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    "API_URL_OPENAI": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    "OPENAI_WEBSOCKET_BASE_URL": os.environ.get("OPENAI_WEBSOCKET_BASE_URL", ""),
    "OPENAI_USE_WEBSOCKET": os.environ.get("OPENAI_USE_WEBSOCKET", "1"),
    "MODEL_DICT": {
        "qwen": model_qwen,
        "openai": model_openai,
        "glmv": model_glmv,
        "claude": model_claude,
    },
    "TRELLIS_SERVER_URL": f"http://localhost:{os.environ.get('SAM3D_PORT', '8080')}",
    "FLUX_SERVER_URL": "",
    "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY_EFFECTIVE", ""),
}
path = pathlib.Path("${SAGE_DIR}/server/key.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(server_key, indent=4) + "\n")
path.chmod(0o600)
print(f"  wrote {path}")
PYEOF
    log "key.json files written."
else
    log "WARNING: OPENAI_API_KEY/OPENROUTER_API_KEY/API_TOKEN not set. key.json not written."
    log "  Set it before running SAGE: export OPENAI_API_KEY=... or OPENROUTER_API_KEY=..."
fi

# ── 1b. Fix spconv if needed (cu120 → cu124 for CUDA 12.4) ────────────────
SAGE_PYTHON="/workspace/miniconda3/envs/sage/bin/python"
if [[ -x "${SAGE_PYTHON}" ]]; then
    _spconv_ver=$("${SAGE_PYTHON}" -c "import spconv; print(spconv.__version__)" 2>/dev/null || echo "missing")
    if [[ "${_spconv_ver}" != "2.3.8" ]]; then
        log "Upgrading spconv (${_spconv_ver} → 2.3.8 for cu124)..."
        "${SAGE_PYTHON}" -m pip install -q spconv-cu124==2.3.8 2>/dev/null && \
            log "spconv upgraded to 2.3.8" || \
            log "WARNING: spconv upgrade failed (SAM3D may crash)"
    else
        log "spconv 2.3.8 OK"
    fi
fi

# ── 2. Apply SAGE patches (idempotent) ──────────────────────────────────────
PATCH_SCRIPT="${WORKSPACE}/BlueprintPipeline/scripts/runpod_sage/apply_sage_patches.sh"
if [[ "${SKIP_PATCHES}" == "1" ]]; then
    log "Skipping SAGE patches (SKIP_PATCHES=1)"
elif [[ -f "${PATCH_SCRIPT}" ]]; then
    log "Applying SAGE patches..."
    bash "${PATCH_SCRIPT}" || log "WARNING: Some patches may have failed (non-fatal)"
    # Validate syntax of patched files
    SYNTAX_OK=1
    for _pf in "${SAGE_DIR}/server/layout.py" "${SAGE_DIR}/server/layout_wo_robot.py" "${SAGE_DIR}/server/vlm.py"; do
        if [[ -f "${_pf}" ]]; then
            if ! "${SAGE_PYTHON}" -c "import py_compile; py_compile.compile('${_pf}', doraise=True)" 2>/dev/null; then
                log "ERROR: Syntax error in ${_pf} after patching!"
                SYNTAX_OK=0
            fi
        fi
    done
    if [[ "${SYNTAX_OK}" != "1" ]]; then
        log "FATAL: One or more patched files have syntax errors."
        exit 1
    fi
else
    log "Skipping SAGE patches (${PATCH_SCRIPT} not found)"
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
_sam3d_modules_missing() {
    local py="$1"
    shift
    local out=""
    local status=0
    if ! out=$("${py}" -c "import importlib.util, sys; mods = sys.argv[1:]; missing = [m for m in mods if importlib.util.find_spec(m) is None]; print(' '.join(missing)); raise SystemExit(0 if not missing else 1)" "$@"); then
        status=$?
    fi
    if [[ "${status}" == "0" ]]; then
        return 0
    fi
    if [[ -n "${out}" ]]; then
        echo "${out}"
    fi
    return 1
}

_sam3d_install_modules() {
    local py="$1"
    "${py}" -m pip install --disable-pip-version-check flask flask-cors psutil requests
}

_sam3d_resolve_python() {
    local -a _py_candidates=()
    local -a _mods=(flask flask_cors psutil requests)
    local py
    local missing
    local probe_py=""

    [[ -n "${SAM3D_PYTHON_BIN:-}" ]] && _py_candidates+=("${SAM3D_PYTHON_BIN}")
    [[ -x "/workspace/miniconda3/envs/sage/bin/python" ]] && _py_candidates+=("/workspace/miniconda3/envs/sage/bin/python")
    [[ -x "/workspace/miniconda3/envs/sage/bin/python3" ]] && _py_candidates+=("/workspace/miniconda3/envs/sage/bin/python3")
    [[ -x "/workspace/miniconda3/bin/python" ]] && _py_candidates+=("/workspace/miniconda3/bin/python")
    [[ -x /usr/bin/python3.11 ]] && _py_candidates+=("/usr/bin/python3.11")
    [[ -x /usr/bin/python3 ]] && _py_candidates+=("/usr/bin/python3")

    local -a _seen=()
    local -A _uniq=()
    for py in "${_py_candidates[@]}"; do
        [[ -z "${py}" ]] && continue
        [[ -x "${py}" ]] || continue
        if [[ -n "${_uniq[${py}]:-}" ]]; then
            continue
        fi
        _seen+=("${py}")
        _uniq["${py}"]=1
    done

    for py in "${_seen[@]}"; do
        if missing=$(_sam3d_modules_missing "${py}" "${_mods[@]}"); then
            echo "${py}"
            return 0
        fi
        if [[ -z "${probe_py}" ]]; then
            probe_py="${py}"
        fi
        log "SAM3D candidate ${py} missing modules: ${missing}"
    done

    if [[ -z "${probe_py}" ]]; then
        return 1
    fi
    if [[ "${SAM3D_AUTO_INSTALL_DEPS}" != "1" ]]; then
        return 1
    fi
    log "Attempting to install required SAM3D Python modules into ${probe_py}..."
    if ! _sam3d_install_modules "${probe_py}"; then
        return 1
    fi
    if _sam3d_modules_missing "${probe_py}" "${_mods[@]}"; then
        echo "${probe_py}"
        return 0
    fi
    return 1
}

if [[ "${SKIP_SAM3D:-0}" != "1" ]]; then
    SAM3D_SCRIPT="${WORKSPACE}/BlueprintPipeline/scripts/runpod_sage/sam3d_server.py"
    if [[ ! -f "${SAM3D_SCRIPT}" ]]; then
        log "ERROR: sam3d_server.py not found at ${SAM3D_SCRIPT}"
    else
        log "Starting SAM3D server on :${SAM3D_PORT} (backend=${SAM3D_IMAGE_BACKEND})..."

        SAM3D_ARGS="--port ${SAM3D_PORT} --image-backend ${SAM3D_IMAGE_BACKEND}"
        [[ -n "${OPENAI_API_KEY_EFFECTIVE:-}" ]] && SAM3D_ARGS="${SAM3D_ARGS} --openai-key ${OPENAI_API_KEY_EFFECTIVE}"
        [[ -n "${GEMINI_API_KEY_EFFECTIVE:-}" ]] && SAM3D_ARGS="${SAM3D_ARGS} --gemini-key ${GEMINI_API_KEY_EFFECTIVE}"

        # Check for SAM3D checkpoints
        if [[ -f "/workspace/sam3d/checkpoints/hf/pipeline.yaml" ]]; then
            SAM3D_ARGS="${SAM3D_ARGS} --checkpoint-dir /workspace/sam3d/checkpoints/hf"
        else
            log "WARNING: SAM3D checkpoints not found at /workspace/sam3d/checkpoints/hf"
            log "  Download with: huggingface-cli download facebook/sam-3d-objects --local-dir /workspace/sam3d/checkpoints/hf"
        fi

        # Prefer conda env if it has SAM3D deps (portable for snapshot-based images).
        if ! SAM3D_CMD_PATH="$(_sam3d_resolve_python)"; then
            log "ERROR: No SAM3D-capable Python interpreter found."
            log "Set SAM3D_PYTHON_BIN to a usable interpreter and ensure Flask/psutil/requests are installed."
            exit 1
        fi
        export SAM3D_PYTHON_BIN="${SAM3D_CMD_PATH}"
        SAM3D_CMD=("${SAM3D_CMD_PATH}")

        nohup "${SAM3D_CMD[@]}" "${SAM3D_SCRIPT}" ${SAM3D_ARGS} \
            > /tmp/sam3d_server.log 2>&1 &
        SAM3D_PID=$!
        echo "${SAM3D_PID}" > /tmp/sam3d.pid
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
export ISAAC_ASSETS_ROOT="${ISAAC_ASSETS_ROOT:-/workspace/isaacsim_assets/Assets/Isaac/5.1}"
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

ISAAC_PREFLIGHT="${WORKSPACE}/BlueprintPipeline/scripts/runpod_sage/isaacsim_runtime_preflight.sh"
if [[ "${SKIP_ISAAC_SIM:-0}" != "1" ]] && [[ -x "${ISAAC_PREFLIGHT}" ]]; then
    log "Running Isaac Sim runtime preflight..."
    if ! ISAACSIM_PY="/workspace/isaacsim_env/bin/python3" "${ISAAC_PREFLIGHT}"; then
        log "WARNING: Isaac Sim runtime preflight failed; skipping Isaac Sim startup."
        SKIP_ISAAC_SIM=1
    fi
fi

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
            MCP_EXT_SRC="$(resolve_mcp_extension_src || true)"
            if [[ -n "${MCP_EXT_SRC}" ]]; then
                ln -sf "${MCP_EXT_SRC}" "${ISAACSIM_PATH}/exts/isaac.sim.mcp_extension"
                log "  MCP extension: ${MCP_EXT_SRC}"
            else
                log "WARNING: MCP extension not found under ${SAGE_DIR}/server/{isaacsim_mcp_ext,isaacsim}"
            fi

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
                nohup /workspace/isaacsim_env/bin/python3 -P -m isaacsim \
                    "${KIT_FILE}" \
                    --no-window \
                    --enable isaac.sim.mcp_extension \
                    > /tmp/isaacsim.log 2>&1 &
            fi
            ISAAC_PID=$!
            echo "${ISAAC_PID}" > /tmp/isaacsim.pid
            log "Isaac Sim PID: ${ISAAC_PID}"

            # Compute expected MCP port
            MCP_PORT=$(python3.11 -c "
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

# ── 6. Start interactive backends (if installed) ─────────────────────────────
INTERACTIVE_BACKENDS_SCRIPT="${WORKSPACE}/BlueprintPipeline/scripts/runpod_sage/start_interactive_backends.sh"
if [[ "${SKIP_INTERACTIVE_BACKENDS:-0}" != "1" ]] && [[ -f "${INTERACTIVE_BACKENDS_SCRIPT}" ]]; then
    # Only start if weights are actually baked in (install_interactive_backends.sh was run)
    if [[ -d "${WORKSPACE}/PhysX-Anything/pretrain/vlm" ]] || [[ -d "${WORKSPACE}/infinigen" ]]; then
        log "Starting interactive asset backends..."
        bash "${INTERACTIVE_BACKENDS_SCRIPT}" all 2>&1 || log "WARNING: Some interactive backends failed to start (non-fatal)"
    else
        log "Interactive backends not installed (run install_interactive_backends.sh to add them)"
    fi
else
    log "Skipping interactive backends (SKIP_INTERACTIVE_BACKENDS=1 or script missing)"
fi

# ── 7. Final status ─────────────────────────────────────────────────────────
log "=========================================="
log "Startup complete."
log ""
log "Services:"
SAM3D_HEALTH=$(curl -sf "http://127.0.0.1:${SAM3D_PORT}/health" 2>/dev/null || echo "not running")
log "  SAM3D:    http://0.0.0.0:${SAM3D_PORT} (${SAM3D_HEALTH})"
if [[ "${SKIP_ISAAC_SIM:-0}" != "1" ]]; then
    log "  Isaac Sim: MCP port ${MCP_PORT:-unknown}"
fi
if [[ -f /tmp/physx_anything.pid ]] && kill -0 "$(cat /tmp/physx_anything.pid 2>/dev/null)" 2>/dev/null; then
    log "  PhysX-Anything: http://localhost:8083"
fi
if [[ -f /tmp/infinigen.pid ]] && kill -0 "$(cat /tmp/infinigen.pid 2>/dev/null)" 2>/dev/null; then
    log "  Infinigen:      http://localhost:8084"
fi
log ""
log "To run SAGE (scene-only mode):"
log "  source /workspace/miniconda3/etc/profile.d/conda.sh && conda activate sage"
log "  cd /workspace/SAGE/client"
log "  python client_generation_room_desc.py \\"
log "    --room_desc 'A modern living room with a sofa and coffee table' \\"
log "    --server_paths ../server/layout_wo_robot.py"
log ""
log "To run FULL pipeline (7 stages + robot + BP post-processing):"
log "  source /workspace/miniconda3/etc/profile.d/conda.sh && conda activate sage"
log "  ROOM_TYPE=kitchen ROBOT_TYPE=mobile_franka \\"
log "    TASK_DESC='Pick up the mug from the counter and place it on the table' \\"
log "    bash /workspace/BlueprintPipeline/scripts/runpod_sage/run_full_pipeline.sh"
log ""
log "One-shot container mode (run once then exit):"
log "  docker run --gpus all -e RUN_MODE=full_pipeline -e OPENAI_API_KEY=... <image>"
log ""
log "Smoke test (1 pose sample, 1 demo, strict):"
log "  bash /workspace/BlueprintPipeline/scripts/runpod_sage/smoke_full_pipeline.sh"
log ""
log "Logs:"
log "  SAM3D:          tail -f /tmp/sam3d_server.log"
log "  Isaac Sim:      tail -f /tmp/isaacsim.log"
log "  PhysX-Anything: tail -f /tmp/physx_anything_service.log"
log "  Infinigen:      tail -f /tmp/infinigen_service.log"
log "=========================================="

if [[ "${RUN_MODE}" == "services" ]]; then
    exec tail -f /dev/null
elif [[ "${RUN_MODE}" == "full_pipeline" ]]; then
    log "RUN_MODE=full_pipeline — running pipeline once, then exiting."
    # shellcheck disable=SC1091
    source /workspace/miniconda3/etc/profile.d/conda.sh
    conda activate sage
    bash /workspace/BlueprintPipeline/scripts/runpod_sage/run_full_pipeline.sh
    log "Pipeline finished successfully. Exiting."
    exit 0
else
    log "ERROR: Unknown RUN_MODE='${RUN_MODE}' (expected: services|full_pipeline)"
    exit 2
fi
