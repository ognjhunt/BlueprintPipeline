FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
# - libgl1: Required by some image processing libraries (replaces libgl1-mesa-glx in newer Debian)
# - libglib2.0-0: Required by various Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# - numpy: Matrix math and data handling
# - usd-core: OpenUSD Python bindings (pxr module)
# - pygltflib: Parse GLB/GLTF files
# - Pillow: Image processing for textures
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    usd-core==24.8 \
    pygltflib==1.16.1 \
    Pillow==10.3.0

# Copy application code
COPY glb_to_usd.py /app/glb_to_usd.py
COPY build_scene_usd.py /app/build_scene_usd.py
COPY assemble_scene.py /app/assemble_scene.py
COPY run_usd.sh /app/run_usd.sh

# Make scripts executable
RUN chmod +x /app/run_usd.sh

# Set Python path
ENV PYTHONPATH=/app

ENTRYPOINT ["/app/run_usd.sh"]