#!/usr/bin/env bash
set -euo pipefail

# Bootstrap SAGE + Isaac Sim via pip install (no Docker required).
# For use on Vast.ai / environments where Docker-in-Docker is not available.

log() { echo "[sage-isaac-pip $(date -u +%FT%TZ)] $*"; }

WORKSPACE=${WORKSPACE:-/workspace}
SAGE_DIR="${WORKSPACE}/SAGE"
SECRETS_ENV_PATH="${WORKSPACE}/.sage_runpod_secrets.env"
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

if [[ -f "${SECRETS_ENV_PATH}" ]]; then
  # shellcheck disable=SC1090
  source "${SECRETS_ENV_PATH}"
  log "Loaded secrets env: ${SECRETS_ENV_PATH}"
fi

OPENAI_API_KEY_EFFECTIVE="${OPENAI_API_KEY:-${OPENROUTER_API_KEY:-}}"
if [[ -z "${OPENAI_API_KEY_EFFECTIVE}" ]]; then
  log "ERROR: OPENAI_API_KEY or OPENROUTER_API_KEY is required."
  exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  log "ERROR: SLURM_JOB_ID is required (used by SAGE MCP port hashing)."
  exit 2
fi
if [[ -z "${TRELLIS_SERVER_URL:-}" ]]; then
  log "WARNING: TRELLIS_SERVER_URL not set yet — will need to be updated before running SAGE."
fi

OPENAI_BASE_URL="${OPENAI_BASE_URL:-${OPENROUTER_BASE_URL:-https://api.openai.com/v1}}"
OPENAI_WEBSOCKET_BASE_URL="${OPENAI_WEBSOCKET_BASE_URL:-}"
if [[ -z "${OPENAI_WEBSOCKET_BASE_URL}" && "${OPENAI_BASE_URL,,}" == *"api.openai.com"* ]]; then
  OPENAI_WEBSOCKET_BASE_URL="wss://api.openai.com/ws/v1/realtime?provider=openai"
fi
OPENAI_USE_WEBSOCKET="${OPENAI_USE_WEBSOCKET:-1}"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-5.1}"
OPENAI_MODEL_QWEN="${OPENAI_MODEL_QWEN:-qwen/qwen3.5-397b-a17b}"
OPENAI_MODEL_OPENAI="${OPENAI_MODEL_OPENAI:-moonshotai/kimi-k2.5}"
OPENAI_MODEL_GLMV="${OPENAI_MODEL_GLMV:-${OPENAI_MODEL}}"
OPENAI_MODEL_CLAUDE="${OPENAI_MODEL_CLAUDE:-${OPENAI_MODEL}}"
export OPENAI_API_KEY_EFFECTIVE
export OPENAI_BASE_URL
export OPENAI_WEBSOCKET_BASE_URL
export OPENAI_USE_WEBSOCKET

log "Preflight: GPU visibility"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -n 1 || true

DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [[ "${DRIVER_MAJOR}" -lt 560 ]]; then
  log "ERROR: NVIDIA driver >= 560 required for Isaac Sim 5.1.0. Found: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  exit 3
fi

log "Installing system dependencies"
apt-get update -qq
apt-get install -y -qq \
  ca-certificates curl wget git jq gnupg \
  libgl1 libglib2.0-0 libxrender1 libxi6 libxxf86vm1 \
  libxfixes3 libxkbcommon0 libsm6 libice6 libxext6 \
  libxrandr2 libxcursor1 libxinerama1 libepoxy0 libxt6 \
  libglu1-mesa libegl1 libopengl0 libglx-mesa0 libvulkan1 vulkan-tools \
  psmisc iproute2 >/dev/null

# ── Clone SAGE ──────────────────────────────────────────────────
log "Cloning NVlabs/sage into ${SAGE_DIR} (if needed)"
if [[ ! -d "${SAGE_DIR}/.git" ]]; then
  git clone --depth 1 https://github.com/NVlabs/sage.git "${SAGE_DIR}"
fi

# ── Install Miniconda for SAGE client env ───────────────────────
MINICONDA_DIR="${WORKSPACE}/miniconda3"
log "Installing Miniconda under ${MINICONDA_DIR} (if needed)"
if [[ ! -x "${MINICONDA_DIR}/bin/conda" ]]; then
  cd "${WORKSPACE}"
  wget -q -O Miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3.sh -b -p "${MINICONDA_DIR}"
  rm -f Miniconda3.sh
fi

# shellcheck disable=SC1091
source "${MINICONDA_DIR}/etc/profile.d/conda.sh"

log "Creating conda env 'sage' from SAGE client/environment.yml (if needed)"
if ! conda env list | awk '{print $1}' | grep -qx sage; then
  conda env create -n sage -f "${SAGE_DIR}/client/environment.yml"
fi

conda activate sage

# ── Write SAGE key.json files ───────────────────────────────────
log "Writing SAGE client key.json"
python3 - <<PY
import json, os, pathlib
client_key = {
    "API_TOKEN": os.environ["OPENAI_API_KEY_EFFECTIVE"],
    "API_URL_QWEN": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    "MODEL_NAME": os.environ.get("OPENAI_MODEL_QWEN", "qwen/qwen3.5-397b-a17b"),
}
path = pathlib.Path("${SAGE_DIR}/client/key.json")
path.write_text(json.dumps(client_key, indent=4) + "\n", encoding="utf-8")
path.chmod(0o600)
print("wrote", str(path))
PY

log "Writing SAGE server key.json"
python3 - <<PY
import json, os, pathlib
model_qwen = os.environ.get("OPENAI_MODEL_QWEN", "qwen/qwen3.5-397b-a17b")
model_openai = os.environ.get("OPENAI_MODEL_OPENAI", "moonshotai/kimi-k2.5")
model_glmv = os.environ.get("OPENAI_MODEL_GLMV", os.environ.get("OPENAI_MODEL", "gpt-5.1"))
model_claude = os.environ.get("OPENAI_MODEL_CLAUDE", os.environ.get("OPENAI_MODEL", "gpt-5.1"))
server_key = {
    "ANTHROPIC_API_KEY": "",
    "API_TOKEN": os.environ["OPENAI_API_KEY_EFFECTIVE"],
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
    "TRELLIS_SERVER_URL": os.environ.get("TRELLIS_SERVER_URL", ""),
    "FLUX_SERVER_URL": "",
}
path = pathlib.Path("${SAGE_DIR}/server/key.json")
path.write_text(json.dumps(server_key, indent=4) + "\n", encoding="utf-8")
path.chmod(0o600)
print("wrote", str(path))
PY

# ── Install Isaac Sim via pip ───────────────────────────────────
log "Creating Isaac Sim Python venv"
python3.11 -m venv "${WORKSPACE}/isaacsim_env"
source "${WORKSPACE}/isaacsim_env/bin/activate"
pip install --upgrade pip -q

log "Installing Isaac Sim 5.1.0 via pip (this takes 10-20 minutes)..."
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com 2>&1 | tail -5

if [[ -x "${PREFLIGHT_SCRIPT}" ]]; then
  log "Running Isaac Sim runtime preflight (Vulkan + import safety)..."
  ISAACSIM_PY="${WORKSPACE}/isaacsim_env/bin/python3" REQUIRE_LOCAL_ROBOT_ASSET="${REQUIRE_LOCAL_ROBOT_ASSET:-0}" "${PREFLIGHT_SCRIPT}"
fi

ISAACSIM_PATH=$("${WORKSPACE}/isaacsim_env/bin/python3" -P -c "import os, pathlib, sys; blocked=pathlib.Path('${SAGE_DIR}/server').resolve(); sys.path=[p for p in sys.path if p and pathlib.Path(p).resolve()!=blocked]; import isaacsim; print(os.path.dirname(isaacsim.__file__))")
log "Isaac Sim installed at: ${ISAACSIM_PATH}"

# ── Set up MCP extension and launch Isaac Sim headless ──────────
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y
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

log "Symlinking SAGE MCP extension into Isaac Sim exts"
MCP_EXT_SRC="$(resolve_mcp_extension_src || true)"
if [[ -n "${MCP_EXT_SRC}" ]]; then
  ln -sf "${MCP_EXT_SRC}" "${ISAACSIM_PATH}/exts/isaac.sim.mcp_extension"
else
  log "WARNING: MCP extension not found under ${SAGE_DIR}/server/{isaacsim_mcp_ext,isaacsim}"
fi

# Find the correct .kit experience file
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
  log "WARNING: Could not find standard .kit file. Searching..."
  KIT_FILE=$(find "${ISAACSIM_PATH}/apps" -name "*.kit" -print -quit 2>/dev/null || true)
  if [[ -z "${KIT_FILE}" ]]; then
    log "ERROR: No .kit experience file found under ${ISAACSIM_PATH}/apps"
    ls -la "${ISAACSIM_PATH}/apps/" 2>/dev/null || true
    exit 5
  fi
fi

log "Launching Isaac Sim headless with MCP extension"
log "  kit binary: ${ISAACSIM_PATH}/kit/kit"
log "  experience: ${KIT_FILE}"
log "  SLURM_JOB_ID: ${SLURM_JOB_ID}"

nohup "${ISAACSIM_PATH}/kit/kit" \
  "${KIT_FILE}" \
  --no-window \
  --enable isaac.sim.mcp_extension \
  > /tmp/isaacsim.log 2>&1 &

ISAAC_PID=$!
log "Isaac Sim PID: ${ISAAC_PID}"

# Compute MCP port from SLURM_JOB_ID (same as SAGE does)
MCP_PORT="$(python3.11 - <<'PY'
import hashlib, os
job_id = os.environ.get("SLURM_JOB_ID", "12345")
port_start, port_end = 8080, 40000
h = int(hashlib.md5(str(job_id).encode()).hexdigest(), 16)
print(port_start + (h % (port_end - port_start + 1)))
PY
)"
log "Expected MCP port: ${MCP_PORT}"

deadline=$(( $(date +%s) + 900 ))
while [[ "$(date +%s)" -lt "${deadline}" ]]; do
  if ss -lnt 2>/dev/null | awk '{print $4}' | grep -q ":${MCP_PORT}$"; then
    log "MCP port ${MCP_PORT} is listening — Isaac Sim is ready!"
    exit 0
  fi
  # Also check if isaac sim process is still alive
  if ! kill -0 "${ISAAC_PID}" 2>/dev/null; then
    log "ERROR: Isaac Sim process died. Last 80 lines of log:"
    tail -n 80 /tmp/isaacsim.log || true
    exit 4
  fi
  sleep 5
done

log "ERROR: MCP port ${MCP_PORT} was not listening within 15 minutes."
log "Isaac Sim log (last 120 lines):"
tail -n 120 /tmp/isaacsim.log || true
exit 4
