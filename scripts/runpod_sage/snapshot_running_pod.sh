#!/usr/bin/env bash
# =============================================================================
# Snapshot a running SAGE + SAM3D pod into a Docker image.
#
# This is the FASTEST way to create a portable image — it captures the exact
# state of your running pod, including all compiled CUDA extensions, downloaded
# checkpoints, applied patches, and installed packages.
#
# Run this script ON THE POD (via SSH).
#
# Usage:
#   bash snapshot_running_pod.sh [docker-hub-username]
#
# What it does:
#   1. Finds the running container ID
#   2. docker commit → creates a local image
#   3. (Optional) docker push to Docker Hub
#
# Prerequisites:
#   - Docker must be available on the host
#   - For push: docker login must be done first
# =============================================================================
set -euo pipefail

log() { echo "[snapshot $(date -u +%FT%TZ)] $*"; }

DOCKER_USER="${1:-}"
IMAGE_NAME="${IMAGE_NAME:-sage-sam3d-hybrid}"
TAG="$(date +%Y%m%d-%H%M%S)"

# ── Helper: tar archive from inside container ────────────────────────────────
create_tar_snapshot() {
    log "Creating tar archive of /workspace..."
    ARCHIVE="/tmp/sage-sam3d-snapshot-$(date +%Y%m%d).tar.gz"

    # Estimate size
    log "Estimating size..."
    TOTAL_SIZE=$(du -sh /workspace 2>/dev/null | cut -f1)
    log "  /workspace total: ${TOTAL_SIZE}"
    log "  Archive will be ~60% of this (gzip compression)"
    log ""

    tar czf "${ARCHIVE}" \
        --exclude='*.log' \
        --exclude='__pycache__' \
        --exclude='.cache/pip' \
        --ignore-failed-read \
        -C / \
        workspace/SAGE \
        workspace/sam3d \
        workspace/miniconda3 \
        workspace/isaacsim_env \
        workspace/BlueprintPipeline/scripts/runpod_sage \
        workspace/BlueprintPipeline/configs \
        opt/scenesmith \
        workspace/entrypoint.sh \
        workspace/apply_sage_patches.sh \
        workspace/.isaacsim_path \
        2>/dev/null

    ARCHIVE_SIZE=$(du -sh "${ARCHIVE}" | cut -f1)
    SHA=$(sha256sum "${ARCHIVE}" | cut -d' ' -f1)
    echo "${SHA}  ${ARCHIVE}" > "${ARCHIVE}.sha256"

    log ""
    log "Archive created: ${ARCHIVE} (${ARCHIVE_SIZE})"
    log "SHA256: ${SHA}"
    log ""
    log "To import on a new machine:"
    log "  1. Copy archive: scp ${ARCHIVE} newmachine:/tmp/"
    log "  2. Extract:      tar xzf /tmp/$(basename ${ARCHIVE}) -C /"
    log "  3. Run:          bash /workspace/entrypoint.sh"
}

# ── 1. Detect environment ────────────────────────────────────────────────────
log "Detecting environment..."

# Check if we're INSIDE a container (most cloud GPU pods run inside Docker)
if [[ -f /.dockerenv ]] || grep -q docker /proc/1/cgroup 2>/dev/null; then
    log "Running INSIDE a container."
    log ""
    log "On Vast.ai/RunPod, you're already inside the container."
    log "To snapshot, run this from the HOST or use the provider's snapshot feature."
    log ""
    log "=== Vast.ai Snapshot ==="
    log "  1. Go to https://cloud.vast.ai/instances/"
    log "  2. Click the gear icon on your instance"
    log "  3. Click 'Create Template' to save as a reusable template"
    log "  OR use the Vast.ai CLI:"
    log "    vastai create template --name '${IMAGE_NAME}' --image-tag 'latest'"
    log ""
    log "=== Alternative: Create tar archive from inside container ==="
    log "This creates a portable snapshot you can import on any machine."
    log ""

    read -rp "Create tar archive? (y/N) " choice
    if [[ "${choice}" =~ ^[Yy] ]]; then
        create_tar_snapshot
    fi
    exit 0
fi

# We're on the host
if ! command -v docker >/dev/null 2>&1; then
    log "ERROR: docker not found. Install Docker first."
    exit 1
fi

# ── 2. Find the running container ───────────────────────────────────────────
log "Finding running container..."
CONTAINER_ID=$(docker ps --format '{{.ID}}\t{{.Image}}' | head -1 | cut -f1)

if [[ -z "${CONTAINER_ID}" ]]; then
    log "ERROR: No running containers found."
    log "  Make sure the pod's container is running."
    exit 2
fi

CONTAINER_IMAGE=$(docker ps --format '{{.Image}}' --filter "id=${CONTAINER_ID}")
log "Found container: ${CONTAINER_ID} (image: ${CONTAINER_IMAGE})"

# ── 3. Show what will be committed ──────────────────────────────────────────
log ""
log "Container contents preview:"
docker exec "${CONTAINER_ID}" du -sh /workspace/SAGE 2>/dev/null || true
docker exec "${CONTAINER_ID}" du -sh /workspace/sam3d 2>/dev/null || true
docker exec "${CONTAINER_ID}" du -sh /workspace/miniconda3 2>/dev/null || true
docker exec "${CONTAINER_ID}" du -sh /workspace/isaacsim_env 2>/dev/null || true
docker exec "${CONTAINER_ID}" du -sh /workspace/BlueprintPipeline 2>/dev/null || true
docker exec "${CONTAINER_ID}" du -sh /opt/scenesmith 2>/dev/null || true
docker exec "${CONTAINER_ID}" du -sh /workspace/scenesmith 2>/dev/null || true
log ""

# ── 4. Docker commit ────────────────────────────────────────────────────────
FULL_TAG="${IMAGE_NAME}:${TAG}"
if [[ -n "${DOCKER_USER}" ]]; then
    FULL_TAG="${DOCKER_USER}/${FULL_TAG}"
fi

log "Committing container ${CONTAINER_ID} → ${FULL_TAG}"
log "This may take 5-15 minutes depending on image size..."

docker commit \
    --change 'ENV DEBIAN_FRONTEND=noninteractive' \
    --change 'ENV PYTHONUNBUFFERED=1' \
    --change 'ENV WORKSPACE=/workspace' \
    --change 'ENV LIDRA_SKIP_INIT=1' \
    --change 'WORKDIR /workspace' \
    --change 'EXPOSE 8080' \
    --change 'CMD ["/workspace/entrypoint.sh"]' \
    --message "SAGE + SAM3D snapshot from running pod ($(date -u +%FT%TZ))" \
    "${CONTAINER_ID}" \
    "${FULL_TAG}"

# Also tag as latest
LATEST_TAG="${IMAGE_NAME}:latest"
if [[ -n "${DOCKER_USER}" ]]; then
    LATEST_TAG="${DOCKER_USER}/${LATEST_TAG}"
fi
docker tag "${FULL_TAG}" "${LATEST_TAG}"

IMAGE_SIZE=$(docker images "${FULL_TAG}" --format '{{.Size}}')
log "Commit complete: ${FULL_TAG} (${IMAGE_SIZE})"
log "Also tagged as: ${LATEST_TAG}"

# ── 5. Push to Docker Hub (optional) ────────────────────────────────────────
if [[ -n "${DOCKER_USER}" ]]; then
    log ""
    read -rp "Push to Docker Hub? (y/N) " push_choice
    if [[ "${push_choice}" =~ ^[Yy] ]]; then
        log "Pushing ${FULL_TAG}..."
        docker push "${FULL_TAG}"
        log "Pushing ${LATEST_TAG}..."
        docker push "${LATEST_TAG}"
        log "Push complete!"
        log ""
        log "Pull on any machine with:"
        log "  docker pull ${LATEST_TAG}"
    fi
fi

log ""
log "=========================================="
log "Snapshot saved: ${FULL_TAG}"
log "=========================================="
log ""
log "Next steps:"
log "  1. Push to registry: docker push ${FULL_TAG}"
log "  2. On new machine:   docker pull ${FULL_TAG}"
log "  3. Run anywhere:"
log "     docker run --gpus all \\"
log "       -e OPENAI_API_KEY=sk-... \\"
log "       -e GEMINI_API_KEY=AIza... \\"
log "       -p 8080:8080 \\"
log "       ${LATEST_TAG}"
