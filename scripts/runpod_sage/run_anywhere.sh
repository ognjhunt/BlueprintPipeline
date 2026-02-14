#!/usr/bin/env bash
# =============================================================================
# Run SAGE + SAM3D on ANY provider using the pre-built Docker image.
#
# This script pulls and runs the Docker image with the correct env vars.
# Works on: Vast.ai, RunPod, Lambda, CoreWeave, AWS, GCP, bare metal.
#
# Usage:
#   bash run_anywhere.sh <registry/image> [options]
#
# Examples:
#   # Run with Docker Hub image
#   bash run_anywhere.sh yourusername/sage-sam3d:latest
#
#   # Run with custom API keys
#   OPENAI_API_KEY=sk-... GEMINI_API_KEY=AIza... \
#     bash run_anywhere.sh yourusername/sage-sam3d:latest
#
#   # Run with env file
#   bash run_anywhere.sh yourusername/sage-sam3d:latest --env-file secrets.env
# =============================================================================
set -euo pipefail

log() { echo "[run $(date -u +%FT%TZ)] $*"; }

IMAGE="${1:-}"
shift || true

if [[ -z "${IMAGE}" ]]; then
    echo "Usage: $0 <docker-image> [--env-file secrets.env]"
    echo ""
    echo "Required env vars (set before running or use --env-file):"
    echo "  OPENAI_API_KEY   — OpenAI API key"
    echo ""
    echo "Recommended env vars:"
    echo "  GEMINI_API_KEY   — Gemini API key (image generation)"
    echo "  SLURM_JOB_ID    — MCP port hash seed (default: 12345)"
    echo ""
    echo "Example:"
    echo "  OPENAI_API_KEY=sk-... GEMINI_API_KEY=AIza... \\"
    echo "    $0 yourusername/sage-sam3d:latest"
    exit 1
fi

# Parse optional args
ENV_FILE=""
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-file)
            ENV_FILE="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="${EXTRA_ARGS} $1"
            shift
            ;;
    esac
done

# ── Verify GPU ───────────────────────────────────────────────────────────────
if command -v nvidia-smi >/dev/null 2>&1; then
    log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
else
    log "WARNING: nvidia-smi not found. SAGE requires a GPU with >= 24GB VRAM."
fi

# ── Verify Docker + NVIDIA runtime ──────────────────────────────────────────
if ! command -v docker >/dev/null 2>&1; then
    log "ERROR: docker not found."
    exit 2
fi

if ! docker info 2>/dev/null | grep -qi nvidia; then
    log "WARNING: NVIDIA Docker runtime may not be installed."
    log "  Install with: distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID)"
    log "  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
fi

# ── Pull image ───────────────────────────────────────────────────────────────
log "Pulling ${IMAGE}..."
docker pull "${IMAGE}"

# ── Build docker run command ─────────────────────────────────────────────────
DOCKER_CMD="docker run -d --gpus all --name sage-sam3d-$(date +%s)"
DOCKER_CMD="${DOCKER_CMD} -p 8080:8080"
DOCKER_CMD="${DOCKER_CMD} --shm-size=16g"

# Add env vars
[[ -n "${OPENAI_API_KEY:-}" ]] && DOCKER_CMD="${DOCKER_CMD} -e OPENAI_API_KEY=${OPENAI_API_KEY}"
[[ -n "${GEMINI_API_KEY:-}" ]] && DOCKER_CMD="${DOCKER_CMD} -e GEMINI_API_KEY=${GEMINI_API_KEY}"
[[ -n "${ANTHROPIC_API_KEY:-}" ]] && DOCKER_CMD="${DOCKER_CMD} -e ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}"
[[ -n "${SLURM_JOB_ID:-}" ]] && DOCKER_CMD="${DOCKER_CMD} -e SLURM_JOB_ID=${SLURM_JOB_ID}"
[[ -n "${OPENAI_MODEL:-}" ]] && DOCKER_CMD="${DOCKER_CMD} -e OPENAI_MODEL=${OPENAI_MODEL}"
[[ -n "${OPENAI_BASE_URL:-}" ]] && DOCKER_CMD="${DOCKER_CMD} -e OPENAI_BASE_URL=${OPENAI_BASE_URL}"
[[ -n "${HF_TOKEN:-}" ]] && DOCKER_CMD="${DOCKER_CMD} -e HF_TOKEN=${HF_TOKEN}"

# Add env file if specified
if [[ -n "${ENV_FILE}" ]]; then
    if [[ -f "${ENV_FILE}" ]]; then
        DOCKER_CMD="${DOCKER_CMD} --env-file ${ENV_FILE}"
    else
        log "WARNING: env file not found: ${ENV_FILE}"
    fi
fi

# EULAs
DOCKER_CMD="${DOCKER_CMD} -e OMNI_KIT_ACCEPT_EULA=YES"
DOCKER_CMD="${DOCKER_CMD} -e ACCEPT_EULA=Y"
DOCKER_CMD="${DOCKER_CMD} -e PRIVACY_CONSENT=Y"

DOCKER_CMD="${DOCKER_CMD} ${EXTRA_ARGS} ${IMAGE}"

log "Starting container..."
log "  ${DOCKER_CMD}"

CONTAINER_ID=$(eval "${DOCKER_CMD}")
CONTAINER_ID_SHORT="${CONTAINER_ID:0:12}"

log "Container started: ${CONTAINER_ID_SHORT}"
log ""
log "Monitor startup:"
log "  docker logs -f ${CONTAINER_ID_SHORT}"
log ""
log "SSH into container:"
log "  docker exec -it ${CONTAINER_ID_SHORT} bash"
log ""
log "Run SAGE (inside container):"
log "  source /workspace/miniconda3/etc/profile.d/conda.sh"
log "  conda activate sage"
log "  cd /workspace/SAGE/client"
log "  python client_generation_room_desc.py \\"
log "    --room_desc 'A modern living room with a sofa and coffee table' \\"
log "    --server_paths ../server/layout_wo_robot.py"
log ""
log "Stop:"
log "  docker stop ${CONTAINER_ID_SHORT}"
