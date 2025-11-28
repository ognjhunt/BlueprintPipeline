#!/bin/bash
set -euo pipefail

# 1) Build the initial scene.usda from layout + scene_assets.json
python build_scene_usd.py

# 2) Optionally convert GLBs to USD(Z) and wire them into the scene.
#    Set GLB_TO_USD_CONVERTER in the Cloud Run job env to enable this.
if [[ -n "${GLB_TO_USD_CONVERTER:-}" ]]; then
  SCENE_PATH="/mnt/gcs/${USD_PREFIX}/scene.usda"
  echo "[USD] Running GLB->USD(Z) conversion and wiring for scene: ${SCENE_PATH}"
  python convert_glb_to_usd_and_rewire.py \
    --scene "${SCENE_PATH}" \
    --converter "${GLB_TO_USD_CONVERTER}" \
    --root "/mnt/gcs"
else
  echo "[USD] GLB_TO_USD_CONVERTER not set; skipping GLB->USD(Z) conversion."
fi
