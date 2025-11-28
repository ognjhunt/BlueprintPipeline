FROM python:3.10-slim

WORKDIR /app

# Python deps:
#  - numpy: used by build_scene_usd.py
#  - usd-core: pxr.Usd / UsdGeom / UsdUtils bindings
#  - pygltflib: can convert .glb -> .gltf (glb2gltf)
#  - Pillow: required by gltf2usd for texture work 
RUN pip install --no-cache-dir \
    numpy \
    usd-core \
    pygltflib \
    Pillow

# Your USD assembly scripts
COPY build_scene_usd.py /app/build_scene_usd.py
COPY convert_glb_to_usd_and_rewire.py /app/convert_glb_to_usd_and_rewire.py
COPY run_usd.sh /app/run_usd.sh

# Third-party converter script (vendored from GitHub)
COPY third_party/gltf2usd.py /app/third_party/gltf2usd.py

# GLB -> USDZ wrapper that our Python script calls
COPY glb_to_usdz.sh /usr/local/bin/glb_to_usdz

RUN chmod +x /app/run_usd.sh /usr/local/bin/glb_to_usdz

ENTRYPOINT ["/app/run_usd.sh"]
