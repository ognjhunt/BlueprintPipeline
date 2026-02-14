#!/usr/bin/env bash
set -eo pipefail

log() { echo "[crane-push $(date -u +%FT%TZ)] $*"; }

ARCHIVE="/workspace/sage-sam3d-snapshot.tar.gz"
BASE_IMAGE="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
TARGET="docker.io/nijelhunt/sage-sam3d:latest"

log "Archive: $(ls -lh ${ARCHIVE} | awk '{print $5}')"
log "Base:    ${BASE_IMAGE}"
log "Target:  ${TARGET}"
log ""
log "Starting crane append + push (this takes 15-40 min for 33GB)..."

/usr/local/bin/crane append \
  --base "${BASE_IMAGE}" \
  --new_layer "${ARCHIVE}" \
  --new_tag "${TARGET}"

log "crane append complete!"
log "Image pushed to: ${TARGET}"
log ""
log "Pull with:"
log "  docker pull nijelhunt/sage-sam3d:latest"
log "DONE"
