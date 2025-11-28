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
#    Then embed buffer data as base64 data URIs so kcoley/gltf2usd can read it.
python - << 'PY' "$GLB"
import base64
import json
import sys
from pathlib import Path
from pygltflib.utils import glb2gltf

glb_path = Path(sys.argv[1])
gltf_path = glb_path.with_suffix(".gltf")

if gltf_path.exists():
    # This is the case that previously caused FileExistsError; now it's OK.
    print(f"[glb2gltf] {gltf_path} already exists; reusing existing file.")
else:
    print(f"[glb2gltf] Converting {glb_path} -> {gltf_path}")
    glb2gltf(str(glb_path))

if not gltf_path.exists():
    raise SystemExit(f"[embed-buffers] Expected GLTF file not found: {gltf_path}")

print(f"[embed-buffers] Embedding binary buffers into {gltf_path} as data URIs...")

# Load glTF JSON
data = json.loads(gltf_path.read_text())

buffers = data.get("buffers", [])
for idx, buf in enumerate(buffers):
    uri = buf.get("uri")
    if not uri:
        continue

    # If it's already a data URI, leave it alone.
    if isinstance(uri, str) and uri.startswith("data:"):
        print(f"[embed-buffers] buffers[{idx}] already uses a data URI; skipping.")
        continue

    # Otherwise, treat it as a file path relative to the .gltf file.
    bin_path = (gltf_path.parent / uri).resolve()
    try:
        raw = bin_path.read_bytes()
    except FileNotFoundError:
        print(f"[embed-buffers] WARNING: buffer file not found: {bin_path}; leaving uri as-is.")
        continue

    b64 = base64.b64encode(raw).decode("ascii")
    buf["uri"] = "data:application/octet-stream;base64," + b64
    print(f"[embed-buffers] Embedded buffer '{uri}' into glTF as data URI ({len(raw)} bytes).")

# Write updated glTF back to disk
gltf_path.write_text(json.dumps(data), encoding="utf-8")
PY

# 2) Convert GLTF -> USDZ using kcoley/gltf2usd
GLTF="${GLB%.glb}.gltf"
GLTF2USD_ROOT="/app/gltf2usd/Source"

echo "[glb_to_usdz] Running gltf2usd on ${GLTF}"

# Ensure Python can find:
#   - the top-level gltf2usd modules (Source)
#   - the internal "_gltf2usd" package (Source/_gltf2usd)
#   - the legacy-style "gltf2" modules (Source/_gltf2usd/gltf2) that use
#     Python 2-style imports like "from Skin import Skin"
export PYTHONPATH="${GLTF2USD_ROOT}:${GLTF2USD_ROOT}/_gltf2usd:${GLTF2USD_ROOT}/_gltf2usd/gltf2:${PYTHONPATH:-}"

# Run from inside the Source directory so relative imports behave as expected
(
  cd "${GLTF2USD_ROOT}"
  python gltf2usd.py \
    -g "${GLTF}" \
    -o "${USDZ}"
)
