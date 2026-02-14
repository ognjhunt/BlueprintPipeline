#!/usr/bin/env bash
# =============================================================================
# Push SAGE + SAM3D Docker image to a registry.
#
# Supports: Docker Hub, GitHub Container Registry (GHCR), and any private registry.
#
# Usage:
#   bash push_docker_image.sh <registry/username>
#
# Examples:
#   bash push_docker_image.sh yourusername              # → Docker Hub
#   bash push_docker_image.sh ghcr.io/yourusername      # → GitHub CR
#   bash push_docker_image.sh 123456.dkr.ecr.us-east-1.amazonaws.com  # → AWS ECR
# =============================================================================
set -euo pipefail

log() { echo "[push $(date -u +%FT%TZ)] $*"; }

REGISTRY="${1:-}"
IMAGE_NAME="sage-sam3d"
TAG="${2:-latest}"

if [[ -z "${REGISTRY}" ]]; then
    echo "Usage: $0 <registry/username> [tag]"
    echo ""
    echo "Examples:"
    echo "  $0 yourusername              # Docker Hub"
    echo "  $0 ghcr.io/yourusername      # GitHub Container Registry"
    exit 1
fi

FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${TAG}"
LATEST_IMAGE="${REGISTRY}/${IMAGE_NAME}:latest"

# Check local image exists
if ! docker images "${IMAGE_NAME}" --format '{{.Tag}}' | grep -q "${TAG}"; then
    # Try with registry prefix
    if ! docker images "${FULL_IMAGE}" --format '{{.Tag}}' | grep -q .; then
        log "ERROR: No local image found for ${IMAGE_NAME}:${TAG}"
        log "  Available images:"
        docker images "${IMAGE_NAME}*" --format '  {{.Repository}}:{{.Tag}} ({{.Size}})'
        exit 2
    fi
fi

# Tag for registry
log "Tagging ${IMAGE_NAME}:${TAG} → ${FULL_IMAGE}"
docker tag "${IMAGE_NAME}:${TAG}" "${FULL_IMAGE}" 2>/dev/null || \
    docker tag "${IMAGE_NAME}:latest" "${FULL_IMAGE}" 2>/dev/null || true

docker tag "${FULL_IMAGE}" "${LATEST_IMAGE}" 2>/dev/null || true

# Check login
log "Checking registry authentication..."
if [[ "${REGISTRY}" == ghcr.io/* ]]; then
    log "GitHub Container Registry detected."
    log "  Login with: echo \$GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin"
elif [[ "${REGISTRY}" == *.dkr.ecr.*.amazonaws.com ]]; then
    log "AWS ECR detected."
    log "  Login with: aws ecr get-login-password | docker login --username AWS --password-stdin ${REGISTRY}"
else
    log "Docker Hub detected."
    log "  Login with: docker login"
fi

# Push
log "Pushing ${FULL_IMAGE}..."
log "  (This may take 30-60 minutes for a ~40GB image on first push)"
docker push "${FULL_IMAGE}"

if [[ "${TAG}" != "latest" ]]; then
    log "Pushing ${LATEST_IMAGE}..."
    docker push "${LATEST_IMAGE}"
fi

log ""
log "=========================================="
log "Push complete!"
log "=========================================="
log ""
log "Pull on any machine:"
log "  docker pull ${LATEST_IMAGE}"
log ""
log "Run on any provider:"
log "  docker run --gpus all \\"
log "    -e OPENAI_API_KEY=sk-... \\"
log "    -e GEMINI_API_KEY=AIza... \\"
log "    -e SLURM_JOB_ID=12345 \\"
log "    -p 8080:8080 \\"
log "    ${LATEST_IMAGE}"
log ""
log "Use as Vast.ai template image:"
log "  ${LATEST_IMAGE}"
