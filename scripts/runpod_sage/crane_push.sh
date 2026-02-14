#!/usr/bin/env bash
set -eo pipefail

log() { echo "[crane-push $(date -u +%FT%TZ)] $*"; }

ARCHIVE="${1:-/workspace/sage-sam3d-snapshot.tar.gz}"
BASE_IMAGE="${BASE_IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
IMAGE_PREFIX="${IMAGE_PREFIX:-docker.io/nijelhunt/sage-sam3d}"
TAG="${2:-latest}"
TARGET="${IMAGE_PREFIX}:${TAG}"
MUTATE="${MUTATE:-1}"

if [[ ! -f "${ARCHIVE}" ]]; then
    log "Archive not found: ${ARCHIVE}"
    log "Usage: $0 /path/to/snapshot.tar.gz [tag]"
    exit 1
fi

log "Archive: $(ls -lh ${ARCHIVE} | awk '{print $5}')"
log "Base:    ${BASE_IMAGE}"
log "Target:  ${TARGET}"
log ""
log "Starting crane append + push..."

/usr/local/bin/crane append \
  --base "${BASE_IMAGE}" \
  --new_layer "${ARCHIVE}" \
  --new_tag "${TARGET}"

log "Crane append complete. Target: ${TARGET}"

if [[ "${MUTATE}" == "1" ]]; then
    log "Applying runtime config mutation (entrypoint/cmd/workdir/expose)..."
    /usr/local/bin/crane mutate \
      --entrypoint '["/workspace/entrypoint.sh"]' \
      --cmd '[]' \
      --workdir /workspace \
      --exposed-ports 8080/tcp \
      "${TARGET}" || {
        log "WARN: crane mutate failed. This is usually due to registry permissions/format."
        log "If needed, re-push via Dockerfile (recommended) and set CMD/EXPOSE in Dockerfile."
    }
fi

if [[ "${TAG}" != "latest" ]]; then
    log "Also tagging/pushing latest from ${TAG}..."
    /usr/local/bin/crane tag "${TARGET}" "${IMAGE_PREFIX}:latest" || true
fi

log ""
log "Done."
log "Pull with:"
log "  docker pull ${IMAGE_PREFIX}:latest"
log "  docker pull ${TARGET}"
