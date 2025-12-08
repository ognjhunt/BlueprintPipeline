# Variation Asset Pipeline - End-to-End SimReady Asset Generation
#
# This job orchestrates the complete variation asset generation pipeline:
# 1. Generate reference images (Gemini 3.0 Pro Image - Nano Banana Pro)
# 2. Convert to 3D (SAM3D or Hunyuan)
# 3. Add physics properties
# 4. Output SimReady USDZ assets
#
# Supports multiple backends:
# - SAM3D: Fast, good for simple objects
# - Hunyuan: High quality, slower
#
# Environment Variables:
#   SCENE_ID: Scene identifier (required)
#   3D_BACKEND: "sam3d" | "hunyuan" | "auto" (default: "auto")
#   QUALITY_MODE: "fast" | "balanced" | "quality" (default: "balanced")
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

# Copy the pipeline script
COPY run_variation_asset_pipeline.py .

# Set environment defaults
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Default quality mode (can be overridden)
ENV QUALITY_MODE=balanced
ENV 3D_BACKEND=auto
ENV SKIP_EXISTING=1

# Entry point
CMD ["python", "run_variation_asset_pipeline.py"]
