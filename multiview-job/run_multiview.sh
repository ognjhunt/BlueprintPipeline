#!/usr/bin/env bash
set -euo pipefail

echo "[MULTIVIEW] Starting run_multiview.sh (Gemini generative mode)"

: "${SEG_PREFIX:?Env SEG_PREFIX is required}"  # e.g. scenes/<sceneId>/seg
: "${MULTIVIEW_PREFIX:?Env MULTIVIEW_PREFIX is required}"  # e.g. scenes/<sceneId>/multiview
: "${BUCKET:?Env BUCKET is required}"

SEG_DIR="/mnt/gcs/${SEG_PREFIX}"
MULTIVIEW_DIR="/mnt/gcs/${MULTIVIEW_PREFIX}"

echo "[MULTIVIEW] BUCKET=${BUCKET}"
echo "[MULTIVIEW] SEG_DIR=${SEG_DIR}"
echo "[MULTIVIEW] MULTIVIEW_DIR=${MULTIVIEW_DIR}"

if [[ ! -d "${SEG_DIR}" ]]; then
  echo "[MULTIVIEW] ERROR: Seg directory not found: ${SEG_DIR}"
  ls -R /mnt/gcs || true
  exit 1
fi

# Check for inventory file
if [[ ! -f "${SEG_DIR}/inventory.json" ]]; then
  echo "[MULTIVIEW] ERROR: inventory.json not found at ${SEG_DIR}/inventory.json"
  exit 1
fi

mkdir -p "${MULTIVIEW_DIR}"

shopt -s nullglob
echo "[MULTIVIEW] Existing multiview object directories:"
existing_mv_dirs=("${MULTIVIEW_DIR}"/obj_*)
if (( ${#existing_mv_dirs[@]} > 0 )); then
  for d in "${existing_mv_dirs[@]}"; do
    echo "  - ${d}"
  done
  echo "[MULTIVIEW] Found existing multiview outputs; skipping generation"
  exit 0
else
  echo "  (none found)"
fi

# Export for Python
export SEG_PREFIX MULTIVIEW_PREFIX BUCKET SCENE_ID

echo "[MULTIVIEW] Generating layout from inventory for SAM3D..."
python /app/generate_layout_from_inventory.py

echo "[MULTIVIEW] Running Gemini generative multiview generation..."
python /app/run_multiview_gemini_generative.py

echo "[MULTIVIEW] Running scene background generation..."
python /app/run_generate_scene_background.py

echo "[MULTIVIEW] Done run_multiview.sh"
