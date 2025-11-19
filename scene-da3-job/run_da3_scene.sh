#!/usr/bin/env bash
set -euo pipefail

echo "[DA3] Starting run_da3_scene.sh"

: "${DATASET_PREFIX:?Env DATASET_PREFIX is required}"   # e.g. scenes/<sceneId>/seg/dataset
: "${OUT_PREFIX:?Env OUT_PREFIX is required}"           # e.g. scenes/<sceneId>/da3
: "${BUCKET:?Env BUCKET is required}"

DATASET_DIR="/mnt/gcs/${DATASET_PREFIX}"
OUT_DIR="/mnt/gcs/${OUT_PREFIX}"
HF_CACHE_DIR="/mnt/gcs/hf-cache"

echo "[DA3] BUCKET=${BUCKET}"
echo "[DA3] DATASET_DIR=${DATASET_DIR}"
echo "[DA3] OUT_DIR=${OUT_DIR}"
echo "[DA3] HF_HOME=${HF_CACHE_DIR}"

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "[DA3] ERROR: Dataset directory not found: ${DATASET_DIR}"
  echo "[DA3] Contents of /mnt/gcs (for debugging):"
  ls -R /mnt/gcs || true
  exit 1
fi

mkdir -p "${OUT_DIR}"
mkdir -p "${HF_CACHE_DIR}"

# Export so Python can read them (HF_* envs are already set in the image)
export DATASET_PREFIX OUT_PREFIX BUCKET SCENE_ID

python /app/run_da3_scene_from_dataset.py

