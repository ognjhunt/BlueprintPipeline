#!/usr/bin/env bash
set -euo pipefail

echo "[HUNYUAN] Starting run_hunyuan.sh"

: "${ASSETS_PREFIX:?Env ASSETS_PREFIX is required}"  # e.g. scenes/<sceneId>/assets
: "${BUCKET:?Env BUCKET is required}"

ASSETS_DIR="/mnt/gcs/${ASSETS_PREFIX}"
HF_CACHE_DIR="/mnt/gcs/hf-cache"

# -------------------------------------------------------------------
# Hugging Face token wiring
# -------------------------------------------------------------------
# You can set either HUGGINGFACE_TOKEN or HF_TOKEN on the Cloud Run job.
# Hunyuan's loader looks specifically for HF_TOKEN when calling snapshot_download,
# so we normalize everything to HF_TOKEN and also set HUGGINGFACE_HUB_TOKEN
# for the usual HF clients.
if [[ -n "${HUGGINGFACE_TOKEN:-}" ]]; then
  export HF_TOKEN="${HUGGINGFACE_TOKEN}"
  export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_TOKEN}"
elif [[ -n "${HF_TOKEN:-}" ]]; then
  # If only HF_TOKEN is set, still expose it for the hub client.
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "[HUNYUAN] HF_TOKEN is set (length=${#HF_TOKEN}). Using authenticated Hugging Face downloads."
else
  echo "[HUNYUAN] WARNING: HF_TOKEN is NOT set. Large Hugging Face model downloads may fail or be rate-limited." >&2
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

# Point HF + Transformers to the GCS-backed cache
export ASSETS_PREFIX BUCKET HF_HOME="${HF_CACHE_DIR}" TRANSFORMERS_CACHE="${HF_CACHE_DIR}"

python /app/run_hunyuan_from_assets.py

echo "[HUNYUAN] Done run_hunyuan.sh"
