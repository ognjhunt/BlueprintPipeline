#!/usr/bin/env bash
set -euo pipefail

echo "[GSAM] Starting run_gsam2.sh"

: "${IMAGES_PREFIX:?Env IMAGES_PREFIX is required}"   # e.g. scenes/... or targets/<sceneId>/frames
: "${SEG_PREFIX:?Env SEG_PREFIX is required}"         # e.g. scenes/.../seg or targets/<sceneId>/seg
: "${BUCKET:?Env BUCKET is required}"

IMAGES_DIR="/mnt/gcs/${IMAGES_PREFIX}"
OUT_DIR="/mnt/gcs/${SEG_PREFIX}"
HF_CACHE_DIR="/mnt/gcs/hf-cache"

echo "[GSAM] BUCKET=${BUCKET}"
echo "[GSAM] IMAGES_DIR=${IMAGES_DIR}"
echo "[GSAM] OUT_DIR=${OUT_DIR}"
echo "[GSAM] HF_HOME=${HF_CACHE_DIR}"

if [[ ! -d "${IMAGES_DIR}" ]]; then
  echo "[GSAM] ERROR: Images directory not found: ${IMAGES_DIR}"
  echo "[GSAM] Contents of /mnt/gcs (for debugging):"
  ls -R /mnt/gcs || true
  exit 1
fi

mkdir -p "${OUT_DIR}"
mkdir -p "${HF_CACHE_DIR}"

# Export so Python can read them
export IMAGES_PREFIX SEG_PREFIX BUCKET
export HF_HOME="${HF_CACHE_DIR}"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}"

python /app/run_gsam2_from_images.py

echo "[GSAM] Done run_gsam2.sh"
