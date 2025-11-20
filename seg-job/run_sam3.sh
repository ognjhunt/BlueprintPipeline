#!/usr/bin/env bash
set -euo pipefail

echo "[SAM3] Starting run_sam3.sh"

: "${IMAGES_PREFIX:?Env IMAGES_PREFIX is required}"   # e.g. scenes/... or targets/<sceneId>/frames
: "${SEG_PREFIX:?Env SEG_PREFIX is required}"         # e.g. scenes/.../seg or targets/<sceneId>/seg
: "${BUCKET:?Env BUCKET is required}"

IMAGES_DIR="/mnt/gcs/${IMAGES_PREFIX}"
OUT_DIR="/mnt/gcs/${SEG_PREFIX}"
HF_CACHE_DIR="/mnt/gcs/hf-cache"

HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN:-${HF_TOKEN:-}}

if [[ -n "${HUGGINGFACE_TOKEN}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_TOKEN}"
fi

echo "[SAM3] BUCKET=${BUCKET}"
echo "[SAM3] IMAGES_DIR=${IMAGES_DIR}"
echo "[SAM3] OUT_DIR=${OUT_DIR}"
echo "[SAM3] HF_HOME=${HF_CACHE_DIR}"

echo "[SAM3] SAM3_PROMPTS=${SAM3_PROMPTS:-<default>}"
echo "[SAM3] SAM3_CONFIDENCE=${SAM3_CONFIDENCE:-0.15}"

if [[ ! -d "${IMAGES_DIR}" ]]; then
  echo "[SAM3] ERROR: Images directory not found: ${IMAGES_DIR}"
  exit 1
fi

mkdir -p "${OUT_DIR}" "${HF_CACHE_DIR}"

export IMAGES_PREFIX SEG_PREFIX BUCKET
export HF_HOME="${HF_CACHE_DIR}"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}"

python /app/run_sam3_from_images.py

echo "[SAM3] Done run_sam3.sh"
