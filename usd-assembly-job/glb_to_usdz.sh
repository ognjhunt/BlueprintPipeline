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

# 1) Convert GLB -> GLTF (same folder, same basename), but be idempotent:
#    if the .gltf already exists, just reuse it instead of failing.
python - << 'PY' "$GLB"
import sys
from pathlib import Path
from pygltflib.utils import glb2gltf

glb_path = Path(sys.argv[1])
gltf_path = glb_path.with_suffix(".gltf")

if gltf_path.exists():
    # This is the case that caused FileExistsError before.
    print(f"[glb2gltf] {gltf_path} already exists; reusing existing file.")
else:
    print(f"[glb2gltf] Converting {glb_path} -> {gltf_path}")
    glb2gltf(str(glb_path))
PY

# 2) Convert GLTF -> USDZ using kcoley/gltf2usd
GLTF="${GLB%.glb}.gltf"
GLTF2USD_PY="/app/gltf2usd/Source/gltf2usd.py"

echo "[glb_to_usdz] Running gltf2usd on ${GLTF}"
python "${GLTF2USD_PY}" \
  -g "${GLTF}" \
  -o "${USDZ}"
