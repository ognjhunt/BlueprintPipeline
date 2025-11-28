FROM python:3.10-slim

WORKDIR /app

# --- System deps (git so we can clone gltf2usd) ---
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# --- Python deps ---
#  - numpy: used by build_scene_usd.py
#  - usd-core: pxr.Usd / UsdGeom / UsdUtils bindings (OpenUSD)
#  - pygltflib: used to convert .glb -> .gltf
#  - Pillow: used by gltf2usd for texture handling
RUN pip install --no-cache-dir \
    numpy \
    usd-core \
    pygltflib \
    Pillow

# --- Clone gltf2usd, but DO NOT install its pinned requirements ---
# Repo: https://github.com/kcoley/gltf2usd
# We rely on the modern numpy/Pillow we just installed instead of the
# very old versions in requirements.txt (which are hard to build on py3.10).
RUN git clone https://github.com/kcoley/gltf2usd.git /app/gltf2usd

# --- Shims to make gltf2usd work under Python 3 ---

# 1) Shim module to satisfy gltf2usd's "import gltf2usdUtils"
#    This file lives in *your* repo and is copied into the cloned gltf2usd
#    Source tree so that gltf2loader.py can import it successfully.
COPY gltf2usdUtils.py /app/gltf2usd/Source/gltf2usdUtils.py

# 2) Shim for the old Python 2 stdlib "sets" module so that
#    `from sets import Set` in gltf2/gltf2/Skin.py works on Python 3.
COPY sets.py /app/gltf2usd/Source/sets.py

# 3) Apply patches to fix compatibility issues with modern USD API and image handling
#    IMPORTANT: Apply these patches BEFORE the unicode replacement to avoid conflicts
COPY gltfimage_fix.patch /tmp/gltfimage_fix.patch
COPY usd_material_api_fix.patch /tmp/usd_material_api_fix.patch
RUN cd /app/gltf2usd && \
    git apply /tmp/gltfimage_fix.patch && \
    git apply /tmp/usd_material_api_fix.patch

# 4) Fix Python 2 'unicode' references in gltf2usd for Python 3 compatibility.
#    In Python 3, 'unicode' doesn't exist - it's just 'str'.
#    We use sed to replace all instances of 'unicode' checks with 'str'.
RUN find /app/gltf2usd -name "*.py" -type f -exec sed -i \
    -e 's/isinstance(\([^,]*\), unicode)/isinstance(\1, str)/g' \
    -e 's/type(\([^)]*\)) == unicode/type(\1) == str/g' \
    -e 's/type(\([^)]*\)) is unicode/type(\1) is str/g' \
    {} +

# --- Your USD assembly scripts ---
COPY build_scene_usd.py /app/build_scene_usd.py
COPY convert_glb_to_usd_and_rewire.py /app/convert_glb_to_usd_and_rewire.py
COPY run_usd.sh /app/run_usd.sh

# --- GLB -> USDZ wrapper that we call from convert_glb_to_usd_and_rewire.py ---
COPY glb_to_usdz.sh /usr/local/bin/glb_to_usdz

RUN chmod +x /app/run_usd.sh /usr/local/bin/glb_to_usdz

ENTRYPOINT ["/app/run_usd.sh"]
