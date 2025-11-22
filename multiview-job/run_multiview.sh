#!/usr/bin/env bash
set -euo pipefail

echo "[MULTIVIEW] Starting run_multiview.sh"

: "${LAYOUT_PREFIX:?Env LAYOUT_PREFIX is required}"        # e.g. scenes/<sceneId>/layout
: "${SEG_DATASET_PREFIX:?Env SEG_DATASET_PREFIX is required}"  # e.g. scenes/<sceneId>/seg/dataset
: "${MULTIVIEW_PREFIX:?Env MULTIVIEW_PREFIX is required}"  # e.g. scenes/<sceneId>/multiview
: "${BUCKET:?Env BUCKET is required}"

LAYOUT_DIR="/mnt/gcs/${LAYOUT_PREFIX}"
SEG_DATASET_DIR="/mnt/gcs/${SEG_DATASET_PREFIX}"
MULTIVIEW_DIR="/mnt/gcs/${MULTIVIEW_PREFIX}"

echo "[MULTIVIEW] BUCKET=${BUCKET}"
echo "[MULTIVIEW] LAYOUT_DIR=${LAYOUT_DIR}"
echo "[MULTIVIEW] SEG_DATASET_DIR=${SEG_DATASET_DIR}"
echo "[MULTIVIEW] MULTIVIEW_DIR=${MULTIVIEW_DIR}"

if [[ ! -d "${LAYOUT_DIR}" ]]; then
  echo "[MULTIVIEW] ERROR: Layout directory not found: ${LAYOUT_DIR}"
  ls -R /mnt/gcs || true
  exit 1
fi

if [[ ! -d "${SEG_DATASET_DIR}" ]]; then
  echo "[MULTIVIEW] ERROR: Seg dataset directory not found: ${SEG_DATASET_DIR}"
  ls -R /mnt/gcs || true
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
  echo "[MULTIVIEW] Found existing multiview outputs; skipping run_multiview_from_layout.py"
  exit 0
else
  echo "  (none found)"
fi

# Export for Python
export LAYOUT_PREFIX SEG_DATASET_PREFIX MULTIVIEW_PREFIX BUCKET SCENE_ID
# Optional: LAYOUT_FILE_NAME (defaults to scene_layout_scaled.json) and VIEWS_PER_OBJECT

python /app/run_multiview_from_layout.py

echo "[MULTIVIEW] Done run_multiview.sh"
