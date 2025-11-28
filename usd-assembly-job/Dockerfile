FROM python:3.10-slim

WORKDIR /app

# numpy: already needed by build_scene_usd.py
# usd-core: provides the pxr OpenUSD Python bindings (Usd, UsdGeom, etc.)
#           per NVIDIA/OpenUSD docs you can install this via pip. :contentReference[oaicite:1]{index=1}
RUN pip install --no-cache-dir numpy usd-core

# Copy your USD assembly scripts into the container
COPY build_scene_usd.py /app/build_scene_usd.py
COPY convert_glb_to_usd_and_rewire.py /app/convert_glb_to_usd_and_rewire.py
COPY run_usd.sh /app/run_usd.sh

RUN chmod +x /app/run_usd.sh

# TODO: If your GLB->USD converter is a standalone binary, copy or install it here,
#       and make sure it's on PATH, e.g.:
#   COPY my_gltf_to_usd /usr/local/bin/my_gltf_to_usd
# or install Omniverse tools in a different base image.

ENTRYPOINT ["/app/run_usd.sh"]
