#!/usr/bin/env bash
set -euo pipefail

echo "[OBJECTS] Starting run_objects.sh"

: "${DA3_PREFIX:?Env DA3_PREFIX is required}"              # e.g. scenes/<sceneId>/da3
: "${SEG_DATASET_PREFIX:?Env SEG_DATASET_PREFIX is required}"  # e.g. scenes/<sceneId>/seg/dataset
: "${LAYOUT_PREFIX:?Env LAYOUT_PREFIX is required}"        # e.g. scenes/<sceneId>/layout
: "${BUCKET:?Env BUCKET is required}"

DA3_DIR="/mnt/gcs/${DA3_PREFIX}"
SEG_DATASET_DIR="/mnt/gcs/${SEG_DATASET_PREFIX}"
LAYOUT_DIR="/mnt/gcs/${LAYOUT_PREFIX}"

echo "[OBJECTS] BUCKET=${BUCKET}"
echo "[OBJECTS] DA3_DIR=${DA3_DIR}"
echo "[OBJECTS] SEG_DATASET_DIR=${SEG_DATASET_DIR}"
echo "[OBJECTS] LAYOUT_DIR=${LAYOUT_DIR}"

if [[ ! -d "${DA3_DIR}" ]]; then
  echo "[OBJECTS] ERROR: DA3 directory not found: ${DA3_DIR}"
  echo "[OBJECTS] Contents of /mnt/gcs:"
  ls -R /mnt/gcs || true
  exit 1
fi

if [[ ! -d "${SEG_DATASET_DIR}" ]]; then
  echo "[OBJECTS] ERROR: Seg dataset directory not found: ${SEG_DATASET_DIR}"
  ls -R /mnt/gcs || true
  exit 1
fi

if [[ ! -d "${LAYOUT_DIR}" ]]; then
  echo "[OBJECTS] ERROR: Layout directory not found: ${LAYOUT_DIR}"
  ls -R /mnt/gcs || true
  exit 1
fi

# Export for Python
export DA3_PREFIX SEG_DATASET_PREFIX LAYOUT_PREFIX BUCKET SCENE_ID

python /app/run_objects_from_layout.py

echo "[OBJECTS] Done run_objects.sh"
