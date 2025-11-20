#!/usr/bin/env bash
set -euo pipefail

echo "[MESHY] Starting run_meshy.sh"

: "${ASSETS_PREFIX:?Env ASSETS_PREFIX is required}"
: "${BUCKET:?Env BUCKET is required}"

ASSETS_DIR="/mnt/gcs/${ASSETS_PREFIX}"
echo "[MESHY] BUCKET=${BUCKET}"
echo "[MESHY] ASSETS_DIR=${ASSETS_DIR}"

if [[ ! -d "${ASSETS_DIR}" ]]; then
  echo "[MESHY] ERROR: assets dir not found: ${ASSETS_DIR}"
  ls -R /mnt/gcs || true
  exit 1
fi

export BUCKET ASSETS_PREFIX SCENE_ID

python /app/run_meshy_from_assets.py

echo "[MESHY] Done run_meshy.sh"
