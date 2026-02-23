#!/usr/bin/env bash
# =============================================================================
# Run SceneSmith with Quality Overrides
# =============================================================================
#
# Wrapper that injects all quality Hydra overrides (ALL_SAM3D, Gemini context
# image, per-asset Gemini images, disabled ArtVIP) into the main.py command.
#
# Usage (inside RunPod pod):
#   bash /workspace/BlueprintPipeline/scripts/run-scenesmith-quality.sh \
#     --prompt "A modern kitchen with marble countertops" \
#     --name kitchen_full
#
# Or with all defaults:
#   bash /workspace/BlueprintPipeline/scripts/run-scenesmith-quality.sh
#
# =============================================================================
set -euo pipefail

# Defaults
PROMPT="A modern kitchen with marble countertops, stainless steel appliances, and a kitchen island with bar stools"
NAME="kitchen_$(date +%Y%m%d_%H%M%S)"
NUM_WORKERS=1
SCENESMITH_DIR="${SCENESMITH_DIR:-/workspace/scenesmith}"
EXTRA_OVERRIDES=()

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt)     PROMPT="$2"; shift 2 ;;
    --name)       NAME="$2"; shift 2 ;;
    --workers)    NUM_WORKERS="$2"; shift 2 ;;
    --dir)        SCENESMITH_DIR="$2"; shift 2 ;;
    --override)   EXTRA_OVERRIDES+=("$2"); shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--prompt TEXT] [--name NAME] [--workers N] [--override KEY=VAL]"
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

# Source environment
[[ -f /workspace/.env ]] && source /workspace/.env

cd "${SCENESMITH_DIR}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate

# Kaolin/timm JIT stability on some CUDA/driver combinations.
export PYTORCH_JIT="${PYTORCH_JIT:-0}"

# Kill zombie processes from previous runs
fuser -k 7005/tcp 7006/tcp 7007/tcp 7008/tcp 7009/tcp 2>/dev/null || true
sleep 2

echo "============================================="
echo "  SceneSmith Quality Run"
echo "============================================="
echo "  Name:    ${NAME}"
echo "  Prompt:  ${PROMPT:0:80}..."
echo "  Workers: ${NUM_WORKERS}"
echo "  Dir:     ${SCENESMITH_DIR}"
echo "============================================="
echo ""

# ---- Quality Hydra overrides ----
# These are the equivalent of:
#   SCENESMITH_PAPER_ALL_SAM3D=true
#   SCENESMITH_PAPER_ENABLE_GEMINI_CONTEXT_IMAGE=true
#   SCENESMITH_PAPER_IMAGE_BACKEND=gemini
#   SCENESMITH_PAPER_ENABLE_FURNITURE_CONTEXT_IMAGE=true
#
# Injected directly as Hydra overrides since main.py doesn't read env vars.

QUALITY_OVERRIDES=()
for prefix in furniture_agent wall_agent ceiling_agent manipuland_agent; do
  QUALITY_OVERRIDES+=(
    "${prefix}.asset_manager.general_asset_source=generated"
    "${prefix}.asset_manager.backend=sam3d"
    "${prefix}.asset_manager.router.strategies.generated.enabled=true"
    "${prefix}.asset_manager.router.strategies.articulated.enabled=false"
    "${prefix}.asset_manager.articulated.sources.partnet_mobility.enabled=false"
    "${prefix}.asset_manager.articulated.sources.artvip.enabled=false"
    "${prefix}.asset_manager.image_generation.backend=gemini"
  )
done

# Furniture agent gets extra context image generation
QUALITY_OVERRIDES+=(
  "furniture_agent.context_image_generation.enabled=true"
)

# Merge extra user overrides
PROMPT_JSON=$(python3 -c 'import json,sys; print(json.dumps([sys.argv[1]]))' "${PROMPT}")
ALL_OVERRIDES=(
  "+name=${NAME}"
  "experiment.prompts=${PROMPT_JSON}"
  "experiment.num_workers=${NUM_WORKERS}"
  "experiment.pipeline.parallel_rooms=false"
  "${QUALITY_OVERRIDES[@]}"
  "${EXTRA_OVERRIDES[@]}"
)

echo "Running with ${#ALL_OVERRIDES[@]} Hydra overrides..."
echo ""

# Run
exec python main.py "${ALL_OVERRIDES[@]}"
