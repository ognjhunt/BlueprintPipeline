#!/usr/bin/env bash
set -euo pipefail

echo "[HUNYUAN] Starting run_hunyuan.sh"

: "${ASSETS_PREFIX:?Env ASSETS_PREFIX is required}"  # e.g. scenes/<sceneId>/assets
: "${BUCKET:?Env BUCKET is required}"

ASSETS_DIR="/mnt/gcs/${ASSETS_PREFIX}"
HF_CACHE_DIR="/mnt/gcs/hf-cache"

HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN:-${HF_TOKEN:-}}
if [[ -n "${HUGGINGFACE_TOKEN}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_TOKEN}"
fi

echo "[HUNYUAN] BUCKET=${BUCKET}"
echo "[HUNYUAN] ASSETS_DIR=${ASSETS_DIR}"
echo "[HUNYUAN] HF_HOME=${HF_CACHE_DIR}"
echo "[HUNYUAN] HUNYUAN_MODEL_PATH=${HUNYUAN_MODEL_PATH:-tencent/Hunyuan3D-2.1}"

if [[ ! -d "${ASSETS_DIR}" ]]; then
  echo "[HUNYUAN] ERROR: assets directory not found: ${ASSETS_DIR}" >&2
  exit 1
fi

mkdir -p "${HF_CACHE_DIR}"

export ASSETS_PREFIX BUCKET HF_HOME="${HF_CACHE_DIR}" TRANSFORMERS_CACHE="${HF_CACHE_DIR}"

python /app/run_hunyuan_from_assets.py

echo "[HUNYUAN] Done run_hunyuan.sh"
