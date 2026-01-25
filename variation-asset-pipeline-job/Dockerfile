# Variation Asset Pipeline - End-to-End SimReady Asset Generation
#
# This job orchestrates the complete variation asset generation pipeline:
# 1. Generate reference images (Gemini 3.0 Pro Image - Nano Banana Pro)
# 2. Convert to 3D (UltraShape)
# 3. Add physics properties
# 4. Output SimReady USDZ assets
#
# 3D Backend: UltraShape
#   Reference: https://github.com/PKU-YuanGroup/UltraShape-1.0
#
# Environment Variables:
#   SCENE_ID: Scene identifier (required)
#   QUALITY_MODE: "fast" | "balanced" | "quality" | "ultra" (default: "balanced")
#   GEMINI_API_KEY: Google AI API key for image generation

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install USD tools for GLB -> USDZ conversion
RUN pip install --no-cache-dir \
    usd-core \
    pygltflib

# Try to install usd_from_gltf (may not be available in all environments)
RUN pip install --no-cache-dir usd-from-gltf || echo "usd-from-gltf not available, USDZ conversion will be limited"

# Clone UltraShape repository
ARG ULTRASHAPE_BRANCH=main
RUN git clone --depth 1 --branch ${ULTRASHAPE_BRANCH} \
    https://github.com/PKU-YuanGroup/UltraShape-1.0.git /app/UltraShape-1.0 || \
    echo "UltraShape clone failed, will skip UltraShape support"

# Install UltraShape dependencies (if repo exists)
RUN if [ -f /app/UltraShape-1.0/requirements.txt ]; then \
        pip install --no-cache-dir -r /app/UltraShape-1.0/requirements.txt || \
        echo "Some UltraShape dependencies failed to install"; \
    fi

# Install cubvh for Marching Cubes acceleration (UltraShape dependency)
RUN pip install --no-cache-dir git+https://github.com/ashawkey/cubvh --no-build-isolation || \
    echo "cubvh installation failed, UltraShape may have reduced performance"

# Install PyTorch3D (UltraShape dependency)
RUN pip install --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt251/download.html || \
    pip install --no-cache-dir pytorch3d || \
    echo "pytorch3d installation failed, UltraShape may not work"

# Copy the ultrashape module from the project
COPY ../ultrashape /app/ultrashape_module/ultrashape
ENV PYTHONPATH="${PYTHONPATH}:/app/ultrashape_module"

# Download UltraShape weights (optional - can be mounted instead)
ARG DOWNLOAD_ULTRASHAPE_WEIGHTS=false
RUN if [ "${DOWNLOAD_ULTRASHAPE_WEIGHTS}" = "true" ]; then \
        mkdir -p /app/ultrashape/checkpoints && \
        python -c "from huggingface_hub import snapshot_download; snapshot_download('infinith/UltraShape', local_dir='/app/ultrashape/checkpoints', allow_patterns=['*.pt', '*.yaml'])" || \
        echo "UltraShape weights download failed, will need to mount weights"; \
    fi

# Copy the pipeline script
COPY run_variation_asset_pipeline.py .

# Set environment defaults
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# UltraShape paths
ENV ULTRASHAPE_REPO_PATH=/app/UltraShape-1.0
ENV ULTRASHAPE_CHECKPOINT=/app/ultrashape/checkpoints/ultrashape_v1.pt
ENV ULTRASHAPE_CONFIG=/app/ultrashape/configs/infer_dit_refine.yaml

# Default quality mode (can be overridden)
ENV QUALITY_MODE=balanced
ENV SKIP_EXISTING=1

# Entry point
CMD ["python", "run_variation_asset_pipeline.py"]
