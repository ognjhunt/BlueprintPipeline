#!/usr/bin/env bash
# bootstrap_trellis_pod.sh — Full TRELLIS setup for a fresh pod.
#
# SMART MODE: If a previous install is detected (conda env 'trellis' exists
# with key packages), skips reinstall and jumps straight to run_trellis.sh.
# This makes pod restarts near-instant instead of 15-20 minutes.
#
# For fresh pods or after importing a snapshot, this handles:
#   1. System deps (libx11, libgl1)
#   2. Clone SAGE repo
#   3. Clone TRELLIS repo
#   4. Install miniconda + trellis conda env
#   5. Install all pip packages (including nvdiffrast with --no-build-isolation)
#   6. Write server.py + central_server.py
#   7. Start server via run_trellis.sh
set -euo pipefail

log() { echo "[trellis-bootstrap $(date -u +%FT%TZ)] $*"; }

WORKSPACE=${WORKSPACE:-/workspace}
SAGE_DIR="${WORKSPACE}/SAGE"
TRELLIS_DIR="${SAGE_DIR}/server/TRELLIS"
LOG_PATH="${WORKSPACE}/trellis_server.log"
SECRETS_ENV_PATH="${WORKSPACE}/.sage_runpod_secrets.env"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${SECRETS_ENV_PATH}" ]]; then
  # shellcheck disable=SC1090
  source "${SECRETS_ENV_PATH}"
  log "Loaded secrets env: ${SECRETS_ENV_PATH}"
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  log "ERROR: HF_TOKEN is required (HuggingFace token for TRELLIS checkpoint download)."
  exit 2
fi

export HF_HOME="${HF_HOME:-${WORKSPACE}/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

log "Preflight: GPU visibility"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  log "ERROR: nvidia-smi not found (GPU not available in this pod)."
  exit 3
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n 1 || true

# ──────────────────────────────────────────────
# FAST PATH: If already fully installed, just start the server
# ──────────────────────────────────────────────
if [[ -d "/root/miniconda3/envs/trellis" ]] && \
   [[ -f "${TRELLIS_DIR}/server.py" ]] && \
   /root/miniconda3/envs/trellis/bin/python -c "import torch, nvdiffrast, flask" 2>/dev/null; then
  log "Existing TRELLIS installation detected — skipping full bootstrap."
  log "Using fast start via run_trellis.sh"

  if [[ -f "${SCRIPT_DIR}/run_trellis.sh" ]]; then
    exec bash "${SCRIPT_DIR}/run_trellis.sh"
  else
    # Fallback: run_trellis.sh might be on the pod already
    REMOTE_RUN="/workspace/BlueprintPipeline/scripts/runpod_sage/run_trellis.sh"
    if [[ -f "${REMOTE_RUN}" ]]; then
      exec bash "${REMOTE_RUN}"
    fi
    log "run_trellis.sh not found — falling through to full bootstrap."
  fi
fi

log "No existing installation found. Running full bootstrap..."

# ──────────────────────────────────────────────
# FULL BOOTSTRAP (only on truly fresh pods)
# ──────────────────────────────────────────────

log "Installing system packages"
apt-get update -qq
apt-get install -y -qq git curl ca-certificates libx11-6 libgl1 >/dev/null

if [[ ! -d "${SAGE_DIR}/.git" ]]; then
  log "Cloning NVlabs/sage into ${SAGE_DIR}"
  git clone --depth 1 https://github.com/NVlabs/sage.git "${SAGE_DIR}"
else
  log "SAGE repo already present"
fi

# Install miniconda (if not present)
CONDA_BASE="/root/miniconda3"
if [[ ! -d "${CONDA_BASE}" ]]; then
  log "Installing Miniconda..."
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "${CONDA_BASE}"
  rm /tmp/miniconda.sh
  "${CONDA_BASE}/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
  "${CONDA_BASE}/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
  "${CONDA_BASE}/bin/conda" init bash 2>/dev/null || true
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Create trellis env (if not present)
if [[ ! -d "${CONDA_BASE}/envs/trellis" ]]; then
  log "Creating conda env 'trellis' (Python 3.10)..."
  conda create -n trellis -y python=3.10
fi

conda activate trellis

# Clone TRELLIS (if not present)
if [[ ! -d "${TRELLIS_DIR}" ]]; then
  log "Cloning TRELLIS..."
  cd "${SAGE_DIR}/server"
  git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git
fi

cd "${TRELLIS_DIR}"

# Install pip packages
log "Installing pip packages (this takes ~10 minutes)..."
pip install -q torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -q spconv-cu120
pip install -q imageio[ffmpeg] easydict rembg numpy==1.26 onnxruntime
pip install -q transformers==4.53.2 xformers==0.0.27.post2
pip install -q open3d plyfile trimesh xatlas pyvista pymeshfix igraph
pip install -q "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"
pip install -q kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html

# CRITICAL: nvdiffrast needs --no-build-isolation
log "Installing nvdiffrast (with --no-build-isolation)..."
pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation

# CRITICAL: diff-gaussian-rasterization also needs --no-build-isolation
log "Installing diff-gaussian-rasterization..."
pip install "git+https://github.com/autonomousvision/mip-splatting.git#subdirectory=submodules/diff-gaussian-rasterization/" --no-build-isolation

pip install -q flask werkzeug psutil flask-cors requests

# Authenticate with HuggingFace
log "Authenticating with HuggingFace..."
huggingface-cli login --token "${HF_TOKEN}" 2>/dev/null || \
  python -c "from huggingface_hub import login; login(token='${HF_TOKEN}')" 2>/dev/null || true

# Write server.py and central_server.py (from upstream start_trellis_server.sh)
# These were originally generated by the upstream script's echo statements.
# We only regenerate them if they don't exist.
if [[ ! -f "${TRELLIS_DIR}/server.py" ]]; then
  log "Generating server.py from upstream start_trellis_server.sh..."
  cd "${SAGE_DIR}/server"
  # Extract just the echo commands to generate server.py and central_server.py
  # This is a one-time operation
  bash -c "
    source ${CONDA_BASE}/etc/profile.d/conda.sh
    conda activate trellis
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
    cd ${TRELLIS_DIR}
    # Just run the start script but kill it after it generates the Python files
    timeout 30 bash ${SAGE_DIR}/server/start_trellis_server.sh &
    sleep 15
    kill %1 2>/dev/null || true
  " 2>/dev/null || true

  if [[ ! -f "${TRELLIS_DIR}/server.py" ]]; then
    log "WARNING: Could not auto-generate server.py. Running full upstream script."
    cd "${SAGE_DIR}/server"
    nohup bash start_trellis_server.sh >"${LOG_PATH}" 2>&1 &
    log "Waiting for TRELLIS (up to 60 minutes)..."
    deadline=$(( $(date +%s) + 3600 ))
    while [[ "$(date +%s)" -lt "${deadline}" ]]; do
      if curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1; then
        log "TRELLIS is healthy"
        exit 0
      fi
      sleep 10
    done
    log "ERROR: TRELLIS did not become healthy within timeout."
    tail -n 80 "${LOG_PATH}" || true
    exit 4
  fi
fi

# ──────────────────────────────────────────────
# Start server using the fast run script
# ──────────────────────────────────────────────
log "Bootstrap complete. Starting server..."
if [[ -f "${SCRIPT_DIR}/run_trellis.sh" ]]; then
  exec bash "${SCRIPT_DIR}/run_trellis.sh"
else
  # Inline start as fallback
  export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
  cd "${TRELLIS_DIR}"
  CUDA_VISIBLE_DEVICES=0 nohup python server.py --port 8081 --gpu 0 > /tmp/trellis_worker.log 2>&1 &
  sleep 5
  nohup python central_server.py "http://localhost:8081" > /tmp/trellis_central.log 2>&1 &

  log "Waiting for TRELLIS health..."
  deadline=$(( $(date +%s) + 300 ))
  while [[ "$(date +%s)" -lt "${deadline}" ]]; do
    if curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1; then
      log "TRELLIS is healthy"
      exit 0
    fi
    sleep 5
  done
  log "ERROR: TRELLIS did not become healthy."
  exit 4
fi
