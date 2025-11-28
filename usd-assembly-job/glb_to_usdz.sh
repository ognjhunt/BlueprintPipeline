#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Usage: glb_to_usdz input.glb output.usdz" >&2
  exit 1
fi

GLB="$1"
USDZ="$2"

echo "[glb_to_usdz] Input GLB:  ${GLB}"
echo "[glb_to_usdz] Output USDZ: ${USDZ}"

# 1) Convert GLB -> GLTF (same folder, same basename)
python - << 'PY' "$GLB"
import sys
from pathlib import Path
from pygltflib.utils import glb2gltf

glb_path = Path(sys.argv[1])
gltf_path = glb_path.with_suffix(".gltf")
print(f"[glb2gltf] Converting {glb_path} -> {gltf_path}")
glb2gltf(str(glb_path))
PY

# 2) Convert GLTF -> USDZ using gltf2usd
GLTF="${GLB%.glb}.gltf"

echo "[glb_to_usdz] Running gltf2usd on ${GLTF}"
python /app/third_party/gltf2usd.py \
  -g "${GLTF}" \
  -o "${USDZ}"
