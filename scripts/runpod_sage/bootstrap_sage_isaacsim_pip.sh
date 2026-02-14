#!/usr/bin/env bash
set -euo pipefail

# Bootstrap SAGE + Isaac Sim via pip install (no Docker required).
# For use on Vast.ai / environments where Docker-in-Docker is not available.

log() { echo "[sage-isaac-pip $(date -u +%FT%TZ)] $*"; }

WORKSPACE=${WORKSPACE:-/workspace}
SAGE_DIR="${WORKSPACE}/SAGE"
SECRETS_ENV_PATH="${WORKSPACE}/.sage_runpod_secrets.env"

if [[ -f "${SECRETS_ENV_PATH}" ]]; then
  # shellcheck disable=SC1090
  source "${SECRETS_ENV_PATH}"
  log "Loaded secrets env: ${SECRETS_ENV_PATH}"
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  log "ERROR: OPENAI_API_KEY is required."
  exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  log "ERROR: SLURM_JOB_ID is required (used by SAGE MCP port hashing)."
  exit 2
fi
if [[ -z "${TRELLIS_SERVER_URL:-}" ]]; then
  log "WARNING: TRELLIS_SERVER_URL not set yet — will need to be updated before running SAGE."
fi

OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-5.1}"

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
  libxrandr2 libxcursor1 libxinerama1 libepoxy0 \
  libglu1-mesa libegl1 libopengl0 libglx-mesa0 \
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
    "API_TOKEN": os.environ["OPENAI_API_KEY"],
    "API_URL_QWEN": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    "MODEL_NAME": os.environ.get("OPENAI_MODEL", "gpt-5.1"),
}
path = pathlib.Path("${SAGE_DIR}/client/key.json")
path.write_text(json.dumps(client_key, indent=4) + "\n", encoding="utf-8")
path.chmod(0o600)
print("wrote", str(path))
PY

log "Writing SAGE server key.json"
python3 - <<PY
import json, os, pathlib
model = os.environ.get("OPENAI_MODEL", "gpt-5.1")
server_key = {
    "ANTHROPIC_API_KEY": "",
    "API_TOKEN": os.environ["OPENAI_API_KEY"],
    "API_URL_QWEN": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    "API_URL_OPENAI": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    "MODEL_DICT": {
        "qwen": model,
        "openai": model,
        "glmv": model,
        "claude": model,
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

ISAACSIM_PATH=$(python3.11 -c "import isaacsim; import os; print(os.path.dirname(isaacsim.__file__))")
log "Isaac Sim installed at: ${ISAACSIM_PATH}"

# ── Set up MCP extension and launch Isaac Sim headless ──────────
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y

log "Symlinking SAGE MCP extension into Isaac Sim exts"
ln -sf "${SAGE_DIR}/server/isaacsim/isaac.sim.mcp_extension" \
       "${ISAACSIM_PATH}/exts/isaac.sim.mcp_extension"

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
