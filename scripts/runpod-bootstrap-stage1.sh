#!/usr/bin/env bash
# =============================================================================
# RunPod Bootstrap for Autonomous Stage 1
# =============================================================================
#
# Called by source-orchestrator.yaml via RUNPOD_BOOTSTRAP_COMMAND.
# Prepares a FRESH RunPod pod for Stage 1 (text-scene-gen + text-scene-adapter)
# using the official SceneSmith paper_stack mode.
#
# This script must be:
#   1. Self-contained (no repo on pod yet — it clones everything)
#   2. Fast (skip what's already installed on /workspace volume)
#   3. Non-interactive (no user input)
#
# The orchestrator passes this as a single command string via SSH, so we
# fetch it from a URL or inline it. Typical usage:
#
#   RUNPOD_BOOTSTRAP_COMMAND="curl -sSfL https://raw.githubusercontent.com/<org>/BlueprintPipeline/main/scripts/runpod-bootstrap-stage1.sh | bash -s"
#
# Or if the repo is already on the volume:
#   RUNPOD_BOOTSTRAP_COMMAND="bash /workspace/BlueprintPipeline/scripts/runpod-bootstrap-stage1.sh"
#
# Environment variables (set by orchestrator):
#   GITHUB_TOKEN          — for private repo clone (optional if public)
#   OPENAI_API_KEY        — required for SceneSmith agents
#   GOOGLE_API_KEY        — required for Gemini context images
#   HF_TOKEN              — required for SAM3D checkpoint download
#
# =============================================================================
set -euo pipefail

log() { echo "[bootstrap $(date +%H:%M:%S)] $*"; }

WORKSPACE=/workspace
BP_DIR="${WORKSPACE}/BlueprintPipeline"
SS_DIR="${WORKSPACE}/scenesmith"
MARKER="${WORKSPACE}/.bootstrap_complete"

# =============================================================================
# Skip if already bootstrapped (persistent volume from previous run)
# =============================================================================
if [[ -f "${MARKER}" ]] && [[ -d "${SS_DIR}/.venv" ]] && [[ -d "${BP_DIR}/.git" ]]; then
  log "Bootstrap marker found — skipping full setup."

  # Re-install system libs (lost on container restart)
  apt-get update -qq 2>/dev/null
  apt-get install -y -qq \
      libxrender1 libxi6 libxxf86vm1 libxfixes3 libgl1 \
      libxkbcommon0 libsm6 libice6 libxext6 libxrandr2 \
      libxcursor1 libxinerama1 libepoxy0 libglu1-mesa \
      libegl1 libegl-mesa0 libgles2-mesa libopengl0 libglx-mesa0 \
      psmisc 2>/dev/null || true
  apt-get remove -y bubblewrap 2>/dev/null || true

  # Re-link uv cache
  if [[ ! -L /root/.cache/uv ]] && [[ -d "${WORKSPACE}/.cache/uv" ]]; then
    rm -rf /root/.cache/uv
    ln -s "${WORKSPACE}/.cache/uv" /root/.cache/uv
  fi

  # Source env
  [[ -f "${WORKSPACE}/.env" ]] && source "${WORKSPACE}/.env"
  export LIDRA_SKIP_INIT=1

  log "Quick re-bootstrap complete."
  exit 0
fi

log "Full bootstrap starting on fresh pod..."

# =============================================================================
# Phase 1: System libraries
# =============================================================================
log "Phase 1: System libraries..."
apt-get update -qq
apt-get install -y -qq \
    libpython3.11-dev \
    libxrender1 libxi6 libxxf86vm1 libxfixes3 libgl1 \
    libxkbcommon0 libsm6 libice6 libxext6 libxrandr2 \
    libxcursor1 libxinerama1 libepoxy0 libglu1-mesa \
    libegl1 libegl-mesa0 libgles2-mesa libopengl0 libglx-mesa0 \
    psmisc git-lfs 2>/dev/null
apt-get remove -y bubblewrap 2>/dev/null || true

# =============================================================================
# Phase 2: uv package manager
# =============================================================================
log "Phase 2: uv..."
if ! command -v uv &>/dev/null && [[ ! -f "$HOME/.local/bin/uv" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

if [[ ! -L /root/.cache/uv ]]; then
  rm -rf /root/.cache/uv
  mkdir -p "${WORKSPACE}/.cache/uv"
  ln -s "${WORKSPACE}/.cache/uv" /root/.cache/uv
fi
export UV_LINK_MODE=copy

# =============================================================================
# Phase 3: Clone repos
# =============================================================================
log "Phase 3: Cloning repos..."

GITHUB_URL_PREFIX="https://github.com"
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  GITHUB_URL_PREFIX="https://${GITHUB_TOKEN}@github.com"
fi

# BlueprintPipeline
if [[ ! -d "${BP_DIR}/.git" ]]; then
  log "  Cloning BlueprintPipeline..."
  git clone --depth 1 "${GITHUB_URL_PREFIX}/nicholashuntdesign/BlueprintPipeline.git" "${BP_DIR}" 2>&1 | tail -3
else
  log "  BlueprintPipeline exists, pulling..."
  cd "${BP_DIR}" && git pull --ff-only origin main 2>/dev/null || true
fi

# SceneSmith (official repo)
if [[ ! -d "${SS_DIR}/.git" ]]; then
  log "  Cloning SceneSmith..."
  git clone --depth 1 "${GITHUB_URL_PREFIX}/nepfaff/scenesmith.git" "${SS_DIR}" 2>&1 | tail -3
else
  log "  SceneSmith exists, pulling..."
  cd "${SS_DIR}" && git pull --ff-only origin main 2>/dev/null || true
fi

cd "${SS_DIR}"
git submodule update --init --recursive 2>/dev/null || true

# =============================================================================
# Phase 4: SceneSmith Python deps
# =============================================================================
log "Phase 4: SceneSmith Python deps..."
cd "${SS_DIR}"
uv sync --no-dev 2>&1 | tail -5
source .venv/bin/activate

# =============================================================================
# Phase 5: GPU packages from source
# =============================================================================
log "Phase 5: GPU packages..."

log "  gsplat..."
uv pip install --no-build-isolation \
    "git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7" \
    2>&1 | tail -3

log "  nvdiffrast..."
uv pip install --no-build-isolation \
    "git+https://github.com/NVlabs/nvdiffrast.git" \
    2>&1 | tail -3

log "  pytorch3d..."
FORCE_CUDA=1 uv pip install --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git" \
    2>&1 | tail -3

log "  kaolin..."
uv pip install kaolin \
    -f "https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html" \
    2>&1 | tail -3

# =============================================================================
# Phase 6: Pin xformers + torch
# =============================================================================
log "Phase 6: Pin xformers + torch..."
uv pip install 'xformers==0.0.28.post3' \
    --index-url https://download.pytorch.org/whl/cu124 \
    2>&1 | tail -3
uv pip install 'torch==2.5.1+cu124' 'torchvision==0.20.1+cu124' \
    --index-url https://download.pytorch.org/whl/cu124 \
    2>&1 | tail -3

# =============================================================================
# Phase 7: SAM3D + remaining deps
# =============================================================================
log "Phase 7: SAM3D deps..."
uv pip install -e external/sam-3d-objects/ --no-deps 2>&1 | tail -2
uv pip install \
    open3d optree roma loguru \
    astor einops-exts point-cloud-utils scikit-image trimesh \
    easydict einops fvcore \
    plyfile spconv-cu120 timm \
    lightning pyvista pymeshfix igraph \
    2>&1 | tail -5
uv pip install \
    'MoGe @ git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b' \
    2>&1 | tail -3

# Flask is needed by scenesmith_service.py / start_text_backend_services.sh
uv pip install flask 2>&1 | tail -2

# =============================================================================
# Phase 8: Environment
# =============================================================================
log "Phase 8: Environment..."
if [[ ! -f "${WORKSPACE}/.env" ]]; then
  cat > "${WORKSPACE}/.env" << 'ENVEOF'
export LIDRA_SKIP_INIT=1
export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1
ENVEOF
  chmod 600 "${WORKSPACE}/.env"
fi

# Append API keys from orchestrator env if available
_append_if_set() {
  local key="$1"
  local val="${!key:-}"
  if [[ -n "${val}" ]] && ! grep -q "^export ${key}=" "${WORKSPACE}/.env" 2>/dev/null; then
    echo "export ${key}=\"${val}\"" >> "${WORKSPACE}/.env"
  fi
}
_append_if_set OPENAI_API_KEY
_append_if_set GOOGLE_API_KEY
_append_if_set HF_TOKEN
_append_if_set OPENROUTER_API_KEY
_append_if_set OPENROUTER_BASE_URL

# Ensure LIDRA_SKIP_INIT
if ! grep -q 'LIDRA_SKIP_INIT' "${WORKSPACE}/.env"; then
  echo 'export LIDRA_SKIP_INIT=1' >> "${WORKSPACE}/.env"
fi

source "${WORKSPACE}/.env"
export LIDRA_SKIP_INIT=1

# =============================================================================
# Phase 9: Download checkpoints
# =============================================================================
log "Phase 9: Checkpoints..."
cd "${SS_DIR}"

if [[ ! -f external/checkpoints/sam3.pt ]] && [[ -n "${HF_TOKEN:-}" ]]; then
  log "  Downloading SAM3D checkpoints..."
  mkdir -p external/checkpoints
  huggingface-cli download facebook/sam3 sam3.pt \
      --local-dir external/checkpoints --token "${HF_TOKEN}" 2>&1 | tail -3
  huggingface-cli download facebook/sam-3d-objects \
      --repo-type model \
      --local-dir external/checkpoints/sam-3d-objects-dl \
      --include 'checkpoints/*' --token "${HF_TOKEN}" 2>&1 | tail -3
  cp external/checkpoints/sam-3d-objects-dl/checkpoints/* external/checkpoints/ 2>/dev/null || true
  rm -rf external/checkpoints/sam-3d-objects-dl
elif [[ -f external/checkpoints/sam3.pt ]]; then
  log "  Checkpoints exist, skipping."
else
  log "  WARNING: No HF_TOKEN, skipping checkpoint download."
fi

# AmbientCG materials
if [[ ! -d data/materials ]] || [[ "$(ls data/materials/ 2>/dev/null | wc -l)" -lt 10 ]]; then
  if [[ -f scripts/download_ambientcg.py ]]; then
    log "  Downloading AmbientCG materials..."
    python3 scripts/download_ambientcg.py --output data/materials -r 1K -f JPG -c 8 2>&1 | tail -5 || \
      log "  WARNING: AmbientCG download failed."
  fi
else
  log "  Materials exist, skipping."
fi

# =============================================================================
# Phase 10: BlueprintPipeline deps (for text-scene-gen and adapter)
# =============================================================================
log "Phase 10: BlueprintPipeline deps..."
cd "${BP_DIR}"
if [[ -f requirements.txt ]]; then
  uv pip install -r requirements.txt 2>&1 | tail -5 || \
    pip install -r requirements.txt 2>&1 | tail -5 || true
fi

# =============================================================================
# Done
# =============================================================================
touch "${MARKER}"
log "Bootstrap complete. Marker written to ${MARKER}."
