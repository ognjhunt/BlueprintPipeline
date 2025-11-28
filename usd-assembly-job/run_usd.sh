#!/bin/bash
set -euo pipefail

echo "=========================================="
echo "USD Assembly Job"
echo "=========================================="
echo "BUCKET:        ${BUCKET:-not set}"
echo "SCENE_ID:      ${SCENE_ID:-not set}"
echo "LAYOUT_PREFIX: ${LAYOUT_PREFIX:-not set}"
echo "ASSETS_PREFIX: ${ASSETS_PREFIX:-not set}"
echo "USD_PREFIX:    ${USD_PREFIX:-not set}"
echo "=========================================="

# Run the main assembly pipeline
python /app/assemble_scene.py

echo "=========================================="
echo "USD Assembly Complete"
echo "=========================================="