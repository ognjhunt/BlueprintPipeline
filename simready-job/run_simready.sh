#!/usr/bin/env bash
set -euo pipefail

echo "[SIMREADY] Starting run_simready.sh"

: "${ASSETS_PREFIX:?Env ASSETS_PREFIX is required}"  # e.g. scenes/<sceneId>/assets
: "${BUCKET:?Env BUCKET is required}"

ASSETS_DIR="/mnt/gcs/${ASSETS_PREFIX}"

echo "[SIMREADY] BUCKET=${BUCKET}"
echo "[SIMREADY] ASSETS_DIR=${ASSETS_DIR}"
echo "[SIMREADY] SIMREADY_PRODUCTION_MODE=${SIMREADY_PRODUCTION_MODE:-}"
echo "[SIMREADY] SIMREADY_ALLOW_HEURISTIC_FALLBACK=${SIMREADY_ALLOW_HEURISTIC_FALLBACK:-}"

if [[ ! -d "${ASSETS_DIR}" ]]; then
  echo "[SIMREADY] ERROR: assets directory not found: ${ASSETS_DIR}" >&2
  exit 1
fi

python /app/prepare_simready_assets.py

echo "[SIMREADY] Done run_simready.sh"
