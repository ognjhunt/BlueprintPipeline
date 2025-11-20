#!/usr/bin/env bash
set -euo pipefail

echo "[SAM3D] Starting run_sam3d.sh"

: "${ASSETS_PREFIX:?Env ASSETS_PREFIX is required}"  # e.g. scenes/<sceneId>/assets
: "${BUCKET:?Env BUCKET is required}"

ASSETS_DIR="/mnt/gcs/${ASSETS_PREFIX}"
HF_CACHE_DIR="/mnt/gcs/hf-cache"

HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN:-${HF_TOKEN:-}}
if [[ -n "${HUGGINGFACE_TOKEN}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_TOKEN}"
fi

echo "[SAM3D] BUCKET=${BUCKET}"
echo "[SAM3D] ASSETS_DIR=${ASSETS_DIR}"
echo "[SAM3D] HF_HOME=${HF_CACHE_DIR}"
echo "[SAM3D] SAM3D_CONFIG_PATH=${SAM3D_CONFIG_PATH:-<default>}"

if [[ ! -d "${ASSETS_DIR}" ]]; then
  echo "[SAM3D] ERROR: assets directory not found: ${ASSETS_DIR}" >&2
  exit 1
fi

mkdir -p "${HF_CACHE_DIR}"

export ASSETS_PREFIX BUCKET HF_HOME="${HF_CACHE_DIR}" TRANSFORMERS_CACHE="${HF_CACHE_DIR}"

python /app/run_sam3d_from_assets.py

echo "[SAM3D] Done run_sam3d.sh"
