#!/usr/bin/env bash
set -euo pipefail

echo "[LAYOUT] Starting run_layout.sh"

: "${DA3_PREFIX:?Env DA3_PREFIX is required}"              # e.g. scenes/<sceneId>/da3
: "${SEG_DATASET_PREFIX:?Env SEG_DATASET_PREFIX is required}"  # e.g. scenes/<sceneId>/seg/dataset
: "${LAYOUT_PREFIX:?Env LAYOUT_PREFIX is required}"        # e.g. scenes/<sceneId>/layout
: "${BUCKET:?Env BUCKET is required}"

DA3_DIR="/mnt/gcs/${DA3_PREFIX}"
SEG_DATASET_DIR="/mnt/gcs/${SEG_DATASET_PREFIX}"
OUT_DIR="/mnt/gcs/${LAYOUT_PREFIX}"

echo "[LAYOUT] BUCKET=${BUCKET}"
echo "[LAYOUT] DA3_DIR=${DA3_DIR}"
echo "[LAYOUT] SEG_DATASET_DIR=${SEG_DATASET_DIR}"
echo "[LAYOUT] OUT_DIR=${OUT_DIR}"

if [[ ! -d "${DA3_DIR}" ]]; then
  echo "[LAYOUT] ERROR: DA3 directory not found: ${DA3_DIR}"
  echo "[LAYOUT] Contents of /mnt/gcs:"
  ls -R /mnt/gcs || true
  exit 1
fi

if [[ ! -d "${SEG_DATASET_DIR}" ]]; then
  echo "[LAYOUT] ERROR: Seg dataset directory not found: ${SEG_DATASET_DIR}"
  ls -R /mnt/gcs || true
  exit 1
fi

mkdir -p "${OUT_DIR}"

# Export for Python
export DA3_PREFIX SEG_DATASET_PREFIX LAYOUT_PREFIX BUCKET SCENE_ID

python /app/run_layout_from_da3.py

echo "[LAYOUT] Done run_layout.sh"
