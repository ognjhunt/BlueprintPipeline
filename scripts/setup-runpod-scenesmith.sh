#!/usr/bin/env bash
# =============================================================================
# SceneSmith + SAM3D Full Bootstrap for RunPod L40S
# =============================================================================
#
# Sets up a FRESH RunPod pod from scratch with everything needed to run
# SceneSmith end-to-end (layout + SAM3D geometry + physics validation).
#
# Base image: runpod-torch template (Python 3.11, torch 2.4.x, CUDA 12.4)
# GPU: NVIDIA L40S (48GB VRAM, Ada Lovelace)
# Cost: ~$0.79/hr
#
# Usage (run INSIDE the RunPod pod via terminal or SSH):
#   bash /workspace/BlueprintPipeline/scripts/setup-runpod-scenesmith.sh
#
# Prerequisites:
#   - /workspace/.env must contain OPENAI_API_KEY (and optionally HF_TOKEN)
#   - SceneSmith repo cloned at /workspace/scenesmith
#   - BlueprintPipeline repo cloned at /workspace/BlueprintPipeline
#
# Estimated time: ~15-20 min (mostly downloading checkpoints)
# =============================================================================
set -euo pipefail

SCRIPT_START=$(date +%s)

log() { echo "[$(date +%H:%M:%S)] $*"; }
fail() { echo "[FAIL] $*" >&2; exit 1; }

run_scenesmith_runtime_patch() {
  local patch_script="/workspace/BlueprintPipeline/scripts/apply_scenesmith_paper_patches.sh"
  if [[ ! -x "${patch_script}" ]]; then
    log "WARNING: SceneSmith runtime patch script missing: ${patch_script}"
    return 0
  fi

  log "Applying SceneSmith runtime patches..."
  if ! SCENESMITH_PAPER_REPO_DIR="/workspace/scenesmith" \
       SCENESMITH_PAPER_PYTHON_BIN="/workspace/scenesmith/.venv/bin/python" \
       "${patch_script}"; then
    log "WARNING: SceneSmith runtime patch step failed (continuing)"
  fi
}

# =============================================================================
# Phase 0: Verify basics
# =============================================================================
log "Phase 0: Preflight checks..."

if [[ ! -d /workspace ]]; then
  fail "/workspace not found. Are you on a RunPod pod with a volume?"
fi

nvidia-smi > /dev/null 2>&1 || fail "No GPU detected"
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
log "GPU: ${GPU_NAME} (${GPU_MEM})"

# =============================================================================
# Phase 1: System libraries
# =============================================================================
log "Phase 1: Installing system libraries..."

apt-get update -qq
apt-get install -y -qq \
    libpython3.11-dev \
    libxrender1 libxi6 libxxf86vm1 libxfixes3 libgl1 \
    libxkbcommon0 libsm6 libice6 libxext6 libxrandr2 \
    libxcursor1 libxinerama1 libepoxy0 libglu1-mesa \
    libegl1 libegl-mesa0 libgles2-mesa libopengl0 libglx-mesa0 \
    psmisc git-lfs 2>/dev/null

# bubblewrap breaks single-GPU Blender server — remove if present
apt-get remove -y bubblewrap 2>/dev/null || true

log "Phase 1 done."

# =============================================================================
# Phase 2: Install uv + redirect cache to workspace volume
# =============================================================================
log "Phase 2: Installing uv package manager..."

if ! command -v uv &>/dev/null && [[ ! -f "$HOME/.local/bin/uv" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# Redirect uv cache to workspace volume (overlay disk is only 20GB)
if [[ ! -L /root/.cache/uv ]]; then
  rm -rf /root/.cache/uv
  mkdir -p /workspace/.cache/uv
  ln -s /workspace/.cache/uv /root/.cache/uv
fi
export UV_LINK_MODE=copy

log "Phase 2 done. uv=$(uv --version 2>/dev/null || echo 'not found')"

# =============================================================================
# Phase 3: Clone repos (if not already present)
# =============================================================================
log "Phase 3: Setting up repositories..."

cd /workspace

if [[ -d scenesmith/.git ]]; then
  log "SceneSmith repo exists, pulling latest..."
  cd scenesmith && git pull origin main 2>/dev/null || true
  cd /workspace
else
  log "SceneSmith repo not found at /workspace/scenesmith."
  log "Clone it manually: git clone <your-scenesmith-repo-url> /workspace/scenesmith"
  log "Then re-run this script."
  fail "Missing /workspace/scenesmith"
fi

# Init submodules (sam-3d-objects lives here)
cd /workspace/scenesmith
git submodule update --init --recursive 2>/dev/null || true

if [[ -d /workspace/BlueprintPipeline/.git ]]; then
  log "BlueprintPipeline repo exists."
else
  log "BlueprintPipeline not found. Clone it to /workspace/BlueprintPipeline."
fi

log "Phase 3 done."

# =============================================================================
# Phase 4: Install SceneSmith core deps
# =============================================================================
log "Phase 4: Installing SceneSmith Python dependencies..."

cd /workspace/scenesmith
uv sync --no-dev 2>&1 | tail -5

source .venv/bin/activate
log "venv activated. Python=$(python3 --version)"

log "Phase 4 done."

# =============================================================================
# Phase 5: Build GPU-accelerated packages (before xformers touches torch)
# =============================================================================
log "Phase 5: Building GPU packages from source..."

# gsplat
log "  Installing gsplat..."
uv pip install --no-build-isolation \
    "git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7" \
    2>&1 | tail -3

# nvdiffrast
log "  Installing nvdiffrast..."
uv pip install --no-build-isolation \
    "git+https://github.com/NVlabs/nvdiffrast.git" \
    2>&1 | tail -3

# pytorch3d (CUDA compilation ~2 min on L40S)
log "  Installing pytorch3d (compiling CUDA kernels, ~2 min)..."
FORCE_CUDA=1 uv pip install --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git" \
    2>&1 | tail -3

# kaolin (must match torch version)
log "  Installing kaolin..."
uv pip install kaolin \
    -f "https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html" \
    2>&1 | tail -3

log "Phase 5 done."

# =============================================================================
# Phase 6: Pin xformers + torch (CRITICAL — xformers pulls torch 2.10 otherwise)
# =============================================================================
log "Phase 6: Pinning xformers + torch versions..."

# xformers 0.0.28.post3 is the last version compatible with torch 2.5.1
uv pip install 'xformers==0.0.28.post3' \
    --index-url https://download.pytorch.org/whl/cu124 \
    2>&1 | tail -3

# Force torch back to 2.5.1 in case xformers pulled a newer version
uv pip install 'torch==2.5.1+cu124' 'torchvision==0.20.1+cu124' \
    --index-url https://download.pytorch.org/whl/cu124 \
    2>&1 | tail -3

log "Phase 6 done."

# =============================================================================
# Phase 7: SAM3D package + remaining deps
# =============================================================================
log "Phase 7: Installing SAM3D and remaining dependencies..."

cd /workspace/scenesmith

# sam3d_objects as editable (--no-deps to avoid pulling extra torch)
uv pip install -e external/sam-3d-objects/ --no-deps 2>&1 | tail -2

# All the deps discovered from SAM3D import errors
uv pip install \
    open3d optree roma loguru \
    astor einops-exts point-cloud-utils scikit-image trimesh \
    easydict einops fvcore \
    plyfile spconv-cu120 'timm>=1.0.25' \
    lightning pyvista pymeshfix igraph utils3d \
    2>&1 | tail -5

# MoGe (pinned commit)
uv pip install \
    'MoGe @ git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b' \
    2>&1 | tail -3

log "Phase 7 done."

# =============================================================================
# Phase 8: Environment variables
# =============================================================================
log "Phase 8: Configuring environment..."

# Create .env if it doesn't exist
if [[ ! -f /workspace/.env ]]; then
  cat > /workspace/.env << 'ENVEOF'
# Required — fill these in:
export OPENAI_API_KEY="<your-openai-api-key>"
# Optional (OpenRouter / OpenAI-compatible gateways):
export OPENROUTER_API_KEY=""
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
# Optional:
export GOOGLE_API_KEY=""
export HF_TOKEN=""

# SAM3D init skip (required — sam3d_objects.__init__ imports missing module)
export LIDRA_SKIP_INIT=1
export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1
export PYTORCH_JIT=0
ENVEOF
  chmod 600 /workspace/.env
  log "Created /workspace/.env — EDIT IT to add your API keys!"
else
  log "/workspace/.env already exists."
  # Ensure LIDRA_SKIP_INIT is present
  if ! grep -q 'LIDRA_SKIP_INIT' /workspace/.env; then
    echo 'export LIDRA_SKIP_INIT=1' >> /workspace/.env
    log "Added LIDRA_SKIP_INIT=1 to .env"
  fi
  if ! grep -q '^export PYTORCH_JIT=' /workspace/.env; then
    echo 'export PYTORCH_JIT=0' >> /workspace/.env
    log "Added PYTORCH_JIT=0 to .env"
  fi
fi

# Auto-source .env from .bashrc
if ! grep -q 'source /workspace/.env' /root/.bashrc 2>/dev/null; then
  echo '[ -f /workspace/.env ] && source /workspace/.env' >> /root/.bashrc
  log "Added .env auto-source to .bashrc"
fi

# Source SceneSmith quality config if present
if [[ -f /workspace/BlueprintPipeline/configs/scenesmith_quality.env ]]; then
  log "SceneSmith quality config found. Adding to .bashrc..."
  if ! grep -q 'scenesmith_quality.env' /root/.bashrc 2>/dev/null; then
    echo '[ -f /workspace/BlueprintPipeline/configs/scenesmith_quality.env ] && set -a && source /workspace/BlueprintPipeline/configs/scenesmith_quality.env && set +a' >> /root/.bashrc
  fi
fi

log "Phase 8 done."

# =============================================================================
# Phase 9: Download checkpoints + data (skip if already present)
# =============================================================================
log "Phase 9: Downloading model checkpoints and data..."

cd /workspace/scenesmith
source /workspace/.env 2>/dev/null || true

# SAM3D checkpoints (~16 GB)
if [[ -f external/checkpoints/sam3.pt ]]; then
  log "  SAM3D checkpoint already exists, skipping."
else
  log "  Downloading SAM3D checkpoints (~16 GB)..."
  mkdir -p external/checkpoints

  if [[ -n "${HF_TOKEN:-}" ]]; then
    huggingface-cli download facebook/sam3 sam3.pt \
        --local-dir external/checkpoints --token "${HF_TOKEN}" 2>&1 | tail -3

    huggingface-cli download facebook/sam-3d-objects \
        --repo-type model \
        --local-dir external/checkpoints/sam-3d-objects-dl \
        --include 'checkpoints/*' --token "${HF_TOKEN}" 2>&1 | tail -3
    cp external/checkpoints/sam-3d-objects-dl/checkpoints/* external/checkpoints/ 2>/dev/null || true
    rm -rf external/checkpoints/sam-3d-objects-dl
  else
    log "  WARNING: HF_TOKEN not set. Skipping checkpoint download."
    log "  Set HF_TOKEN in /workspace/.env and re-run this phase manually."
  fi
fi

# ArtVIP data (~8.7 GB) — NOTE: with ALL_SAM3D=true this is optional
if [[ -d data/artvip_sdf ]] && [[ "$(ls data/artvip_sdf/ 2>/dev/null | wc -l)" -gt 0 ]]; then
  log "  ArtVIP data already exists, skipping."
else
  log "  Downloading ArtVIP data (~8.7 GB)..."
  if [[ -n "${HF_TOKEN:-}" ]]; then
    huggingface-cli download nepfaff/scenesmith-preprocessed-data \
        artvip/artvip_vhacd.tar.gz \
        --repo-type dataset --local-dir . --token "${HF_TOKEN}" 2>&1 | tail -3
    mkdir -p data/artvip_sdf
    tar xzf artvip/artvip_vhacd.tar.gz -C data/artvip_sdf 2>/dev/null || true
    rm -rf artvip
  else
    log "  WARNING: HF_TOKEN not set. Skipping ArtVIP download."
    log "  With SCENESMITH_PAPER_ALL_SAM3D=true, ArtVIP is not needed anyway."
  fi
fi

# AmbientCG materials (~10-15 GB)
if [[ -d data/materials ]] && [[ "$(ls data/materials/ 2>/dev/null | wc -l)" -gt 10 ]]; then
  log "  AmbientCG materials already exist, skipping."
else
  log "  Downloading AmbientCG materials (~10 GB, may take a while)..."
  if [[ -f scripts/download_ambientcg.py ]]; then
    python3 scripts/download_ambientcg.py --output data/materials -r 1K -f JPG -c 8 2>&1 | tail -5 || \
      log "  WARNING: AmbientCG download failed. Textures may be missing."
  else
    log "  WARNING: download_ambientcg.py not found. Skipping materials."
  fi
fi

log "Phase 9 done."

# =============================================================================
# Phase 10: SceneSmith runtime patch pass
# =============================================================================
log "Phase 10: SceneSmith runtime patch pass..."
run_scenesmith_runtime_patch

# =============================================================================
# Phase 11: Verify installation
# =============================================================================
log "Phase 11: Verifying installation..."

cd /workspace/scenesmith
source .venv/bin/activate
source /workspace/.env 2>/dev/null || true

VERIFY_OK=true
python3 -c "
import sys
checks = []
try:
    import torch
    checks.append(f'torch {torch.__version__} cuda={torch.cuda.is_available()}')
    assert torch.cuda.is_available(), 'CUDA not available'
    assert '2.5.1' in torch.__version__, f'Wrong torch version: {torch.__version__}'
except Exception as e:
    checks.append(f'torch FAIL: {e}')
    sys.exit(1)

for name in ['kaolin', 'nvdiffrast', 'pytorch3d', 'gsplat']:
    try:
        __import__(name)
        checks.append(f'{name} OK')
    except Exception as e:
        checks.append(f'{name} FAIL: {e}')

try:
    import xformers
    checks.append(f'xformers {xformers.__version__} OK')
except Exception as e:
    checks.append(f'xformers FAIL: {e}')

try:
    import open3d
    checks.append('open3d OK')
except Exception as e:
    checks.append(f'open3d FAIL: {e}')

try:
    import bpy
    checks.append(f'bpy {bpy.app.version_string} OK')
except Exception as e:
    checks.append(f'bpy FAIL: {e}')

try:
    from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap
    checks.append('SAM3D pipeline OK')
except Exception as e:
    checks.append(f'SAM3D pipeline FAIL: {e}')

for c in checks:
    print(f'  {c}')

failures = [c for c in checks if 'FAIL' in c]
if failures:
    print(f'\n  {len(failures)} FAILURES')
    sys.exit(1)
else:
    print('\n  ALL CHECKS PASSED')
" || VERIFY_OK=false

if [[ "${VERIFY_OK}" == "false" ]]; then
  log "WARNING: Some verification checks failed. See above."
  log "You may need to install missing packages manually."
else
  log "All verification checks passed!"
fi

# =============================================================================
# Done
# =============================================================================
ELAPSED=$(( $(date +%s) - SCRIPT_START ))
MINS=$(( ELAPSED / 60 ))
SECS=$(( ELAPSED % 60 ))

echo ""
echo "============================================="
echo "  SceneSmith Setup Complete (${MINS}m ${SECS}s)"
echo "============================================="
echo ""
echo "  Next steps:"
echo "    1. Edit /workspace/.env with your API keys (if not already done)"
echo "    2. Source the environment:"
echo "         source /workspace/.env"
echo "         cd /workspace/scenesmith && source .venv/bin/activate"
echo ""
echo "    3. Run SceneSmith (with ALL quality overrides baked into Hydra):"
echo ""
echo "         python main.py +name=kitchen_full \\"
echo "           'experiment.prompts=[A modern kitchen with marble countertops and stainless steel appliances]' \\"
echo "           experiment.num_workers=1 \\"
echo "           experiment.pipeline.parallel_rooms=false \\"
echo "           furniture_agent.asset_manager.general_asset_source=generated \\"
echo "           furniture_agent.asset_manager.backend=sam3d \\"
echo "           furniture_agent.asset_manager.router.strategies.generated.enabled=true \\"
echo "           furniture_agent.asset_manager.router.strategies.articulated.enabled=false \\"
echo "           furniture_agent.asset_manager.articulated.sources.partnet_mobility.enabled=false \\"
echo "           furniture_agent.asset_manager.articulated.sources.artvip.enabled=false \\"
echo "           furniture_agent.asset_manager.image_generation.backend=gemini \\"
echo "           furniture_agent.context_image_generation.enabled=true \\"
echo "           wall_agent.asset_manager.general_asset_source=generated \\"
echo "           wall_agent.asset_manager.backend=sam3d \\"
echo "           wall_agent.asset_manager.router.strategies.generated.enabled=true \\"
echo "           wall_agent.asset_manager.router.strategies.articulated.enabled=false \\"
echo "           wall_agent.asset_manager.articulated.sources.partnet_mobility.enabled=false \\"
echo "           wall_agent.asset_manager.articulated.sources.artvip.enabled=false \\"
echo "           wall_agent.asset_manager.image_generation.backend=gemini \\"
echo "           ceiling_agent.asset_manager.general_asset_source=generated \\"
echo "           ceiling_agent.asset_manager.backend=sam3d \\"
echo "           ceiling_agent.asset_manager.router.strategies.generated.enabled=true \\"
echo "           ceiling_agent.asset_manager.router.strategies.articulated.enabled=false \\"
echo "           ceiling_agent.asset_manager.articulated.sources.partnet_mobility.enabled=false \\"
echo "           ceiling_agent.asset_manager.articulated.sources.artvip.enabled=false \\"
echo "           ceiling_agent.asset_manager.image_generation.backend=gemini \\"
echo "           manipuland_agent.asset_manager.general_asset_source=generated \\"
echo "           manipuland_agent.asset_manager.backend=sam3d \\"
echo "           manipuland_agent.asset_manager.router.strategies.generated.enabled=true \\"
echo "           manipuland_agent.asset_manager.router.strategies.articulated.enabled=false \\"
echo "           manipuland_agent.asset_manager.articulated.sources.partnet_mobility.enabled=false \\"
echo "           manipuland_agent.asset_manager.articulated.sources.artvip.enabled=false \\"
echo "           manipuland_agent.asset_manager.image_generation.backend=gemini"
echo ""
echo "    IMPORTANT: The env vars in scenesmith_quality.env are only read by"
echo "    scenesmith_paper_command.py (the BlueprintPipeline bridge). When running"
echo "    main.py directly, you MUST pass the Hydra overrides on the command line"
echo "    (as shown above). The command above includes all overrides."
echo ""
echo "  Kill stuck processes:"
echo "    fuser -k 7005/tcp 7006/tcp 7007/tcp 7008/tcp 7009/tcp 2>/dev/null"
echo "============================================="
