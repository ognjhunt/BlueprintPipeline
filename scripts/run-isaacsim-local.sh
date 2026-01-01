#!/bin/bash
# =============================================================================
# Run Isaac Sim Episode Generation Locally
# =============================================================================
# Convenience script for running episode generation with Isaac Sim locally.
#
# Prerequisites:
#   - NVIDIA GPU with 8GB+ VRAM
#   - Docker with nvidia-docker runtime
#   - Scene data in ./scenes directory
#
# Usage:
#   ./scripts/run-isaacsim-local.sh kitchen_001
#   ./scripts/run-isaacsim-local.sh kitchen_001 --data-pack full --cameras 4
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="blueprint-episode-gen:isaacsim"

# Parse arguments
SCENE_ID="${1:-}"
shift || true

if [ -z "$SCENE_ID" ]; then
    log_error "Usage: $0 <scene_id> [options]"
    echo ""
    echo "Options:"
    echo "  --data-pack <tier>     Data pack tier: core, plus, full (default: core)"
    echo "  --cameras <num>        Number of cameras: 1-4 (default: 1)"
    echo "  --episodes <num>       Episodes per variation (default: 10)"
    echo "  --robot <type>         Robot type: franka, ur10, fetch (default: franka)"
    echo "  --interactive          Run interactive shell instead"
    echo "  --build                Rebuild the Docker image first"
    echo ""
    echo "Examples:"
    echo "  $0 kitchen_001"
    echo "  $0 kitchen_001 --data-pack full --cameras 4"
    echo "  $0 kitchen_001 --interactive"
    exit 1
fi

# Default values
DATA_PACK_TIER="core"
NUM_CAMERAS="1"
EPISODES_PER_VARIATION="10"
ROBOT_TYPE="franka"
INTERACTIVE=false
REBUILD=false

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-pack)
            DATA_PACK_TIER="$2"
            shift 2
            ;;
        --cameras)
            NUM_CAMERAS="$2"
            shift 2
            ;;
        --episodes)
            EPISODES_PER_VARIATION="$2"
            shift 2
            ;;
        --robot)
            ROBOT_TYPE="$2"
            shift 2
            ;;
        --interactive)
            INTERACTIVE=true
            shift
            ;;
        --build)
            REBUILD=true
            shift
            ;;
        *)
            log_warning "Unknown option: $1"
            shift
            ;;
    esac
done

# =============================================================================
# Validation
# =============================================================================

log_info "Validating environment..."

# Check Docker
if ! command -v docker &> /dev/null; then
    log_error "Docker not found. Please install Docker."
    exit 1
fi

# Check NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi not found. NVIDIA GPU and drivers required."
    exit 1
fi

log_info "GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

# Check nvidia-docker
if ! docker info 2>/dev/null | grep -q "nvidia"; then
    log_warning "NVIDIA Docker runtime may not be configured"
    log_info "Trying to run anyway..."
fi

# Check scene directory
SCENE_DIR="${PROJECT_ROOT}/scenes/${SCENE_ID}"
if [ ! -d "$SCENE_DIR" ]; then
    log_warning "Scene directory not found: $SCENE_DIR"
    log_info "Creating directory structure..."
    mkdir -p "$SCENE_DIR/assets"
    mkdir -p "$SCENE_DIR/usd"
fi

# =============================================================================
# Build Image (if needed)
# =============================================================================

if [ "$REBUILD" = true ] || ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
    log_info "Building Docker image: $IMAGE_NAME"
    log_info "This may take 10-20 minutes for first build..."

    docker build \
        -f "${PROJECT_ROOT}/episode-generation-job/Dockerfile.isaacsim" \
        -t "$IMAGE_NAME" \
        "$PROJECT_ROOT"

    log_success "Image built successfully"
else
    log_info "Using existing image: $IMAGE_NAME"
fi

# =============================================================================
# Run Container
# =============================================================================

log_info "=========================================="
log_info "Running Isaac Sim Episode Generation"
log_info "=========================================="
log_info "Scene ID: $SCENE_ID"
log_info "Robot: $ROBOT_TYPE"
log_info "Data Pack: $DATA_PACK_TIER"
log_info "Cameras: $NUM_CAMERAS"
log_info "Episodes/Variation: $EPISODES_PER_VARIATION"
log_info "=========================================="

# Create output directory
OUTPUT_DIR="${PROJECT_ROOT}/output/${SCENE_ID}"
mkdir -p "$OUTPUT_DIR"
mkdir -p "${PROJECT_ROOT}/logs"

# Determine command
if [ "$INTERACTIVE" = true ]; then
    DOCKER_CMD="shell"
    DOCKER_OPTS="-it"
else
    DOCKER_CMD="generate"
    DOCKER_OPTS=""
fi

# Run container
docker run --rm $DOCKER_OPTS \
    --gpus all \
    --shm-size=16g \
    --privileged \
    -e SCENE_ID="$SCENE_ID" \
    -e ROBOT_TYPE="$ROBOT_TYPE" \
    -e DATA_PACK_TIER="$DATA_PACK_TIER" \
    -e NUM_CAMERAS="$NUM_CAMERAS" \
    -e EPISODES_PER_VARIATION="$EPISODES_PER_VARIATION" \
    -e HEADLESS=1 \
    -e USE_LLM="${USE_LLM:-true}" \
    -e USE_CPGEN="${USE_CPGEN:-true}" \
    -e GEMINI_API_KEY="${GEMINI_API_KEY:-}" \
    -v "${PROJECT_ROOT}/scenes:/mnt/local/scenes:rw" \
    -v "${OUTPUT_DIR}:/output:rw" \
    -v "${PROJECT_ROOT}/logs:/logs:rw" \
    "$IMAGE_NAME" \
    "$DOCKER_CMD"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    log_success "Episode generation completed!"
    log_info "Output: $OUTPUT_DIR"
else
    log_error "Episode generation failed with exit code $EXIT_CODE"
    log_info "Check logs at: ${PROJECT_ROOT}/logs"
fi

exit $EXIT_CODE
