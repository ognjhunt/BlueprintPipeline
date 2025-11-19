#!/usr/bin/env bash
set -euo pipefail

echo "[SCALE] Starting run_scale.sh"

: "${LAYOUT_PREFIX:?Env LAYOUT_PREFIX is required}"   # e.g. scenes/<sceneId>/layout
: "${BUCKET:?Env BUCKET is required}"

LAYOUT_DIR="/mnt/gcs/${LAYOUT_PREFIX}"

echo "[SCALE] BUCKET=${BUCKET}"
echo "[SCALE] LAYOUT_DIR=${LAYOUT_DIR}"

if [[ ! -d "${LAYOUT_DIR}" ]]; then
  echo "[SCALE] ERROR: Layout directory not found: ${LAYOUT_DIR}"
  echo "[SCALE] Contents of /mnt/gcs:"
  ls -R /mnt/gcs || true
  exit 1
fi

# Export for Python
export LAYOUT_PREFIX BUCKET SCENE_ID

python /app/run_scale_from_layout.py

echo "[SCALE] Done run_scale.sh"
