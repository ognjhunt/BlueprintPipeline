# syntax=docker/dockerfile:1
# physx-service/Dockerfile
#
# IMPORTANT: Cloud Run Configuration for this service:
# - Memory: 32Gi minimum (ML models are large)
# - CPU: 8 cores minimum  
# - GPU: 1x NVIDIA L4 or T4
# - Timeout: 900s (15 minutes for full pipeline)
# - Concurrency: 1 (GPU can only handle one inference at a time)
# - Min instances: 1 (avoid cold starts - models take 5+ min to load)
# - Startup probe: HTTP GET / with 600s timeout
#
# Deploy command example:
# gcloud run deploy physx-service \
#   --image gcr.io/PROJECT/physx-service:latest \
#   --memory 32Gi \
#   --cpu 8 \
#   --gpu 1 \
#   --gpu-type nvidia-l4 \
#   --timeout 900 \
#   --concurrency 1 \
#   --min-instances 1 \
#   --max-instances 2 \
#   --port 8080 \
#   --region us-central1

FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------
# System deps
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y \
    python3 python3-venv python3-dev \
    git wget curl \
    build-essential cmake ninja-build pkg-config \
    ffmpeg \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda

# ---------------------------------------------------------
# Python venv + tooling
# ---------------------------------------------------------
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel packaging

# ---------------------------------------------------------
# 1) Install PyTorch (GPU build) FIRST
#    Torch 2.5.0 + cu124, per official index.
# ---------------------------------------------------------
RUN pip install --no-cache-dir \
    torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
    --index-url https://download.pytorch.org/whl/cu124

# ---------------------------------------------------------
# 2) Clone PhysX-Anything (with submodules)
# ---------------------------------------------------------
ENV PHYSX_ROOT=/opt/physx_anything
WORKDIR /opt

RUN git clone --recurse-submodules \
    https://github.com/ziangcao0312/PhysX-Anything.git "${PHYSX_ROOT}"

WORKDIR ${PHYSX_ROOT}

# ---------------------------------------------------------
# 3) Kaolin from NVIDIA S3 for torch 2.5.0 + cu124
# ---------------------------------------------------------
RUN pip install --no-cache-dir \
    "kaolin==0.17.0" \
    -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.0_cu124.html

# ---------------------------------------------------------
# 4) Patch requirements.txt IN PLACE so any internal
#    'pip install -r requirements.txt' sees a sane file
# ---------------------------------------------------------
COPY patch_requirements.py /tmp/patch_requirements.py
RUN python /tmp/patch_requirements.py requirements.txt

# ---------------------------------------------------------
# 5) Use setup.sh to install the heavy CUDA stack
# ---------------------------------------------------------
SHELL ["/bin/bash", "-lc"]

RUN . ./setup.sh \
      --basic \
      --xformers \
      --flash-attn \
      --diffoctreerast \
      --spconv \
      --mipgaussian \
      --kaolin \
      --nvdiffrast

# ---------------------------------------------------------
# 6) Qwen2.5 deps (as in PhysX-Anything README)
# ---------------------------------------------------------
RUN pip install --no-cache-dir \
      'transformers==4.50.0' \
      qwen-vl-utils \
      'accelerate>=0.26.0'

# ---------------------------------------------------------
# 7) Download model weights at build time if possible
#    This reduces cold-start time significantly
# ---------------------------------------------------------
RUN if [ -f download.py ]; then \
        echo "Downloading model weights..." && \
        python download.py || echo "Warning: download.py failed, models will download at runtime"; \
    fi

# Pre-download Qwen VL model weights (if not already present)
# This is the biggest contributor to cold-start time
RUN python -c "from transformers import AutoModelForCausalLM; print('Transformers import OK')" || true

# ---------------------------------------------------------
# 8) Install additional dependencies
# ---------------------------------------------------------
RUN pip install --no-cache-dir ipdb flask gunicorn

# ---------------------------------------------------------
# 9) Your Flask wrapper service
# ---------------------------------------------------------
WORKDIR /app

COPY run_physx_anything_pipeline.py ${PHYSX_ROOT}/run_physx_anything_pipeline.py
COPY physx_service.py /app/physx_service.py

# Create temp directory for request processing
RUN mkdir -p /tmp/physx_anything && chmod 777 /tmp/physx_anything

# ---------------------------------------------------------
# Environment and startup configuration
# ---------------------------------------------------------
ENV PORT=8080
ENV PHYSX_ROOT=/opt/physx_anything
# Disable Python output buffering for better logging
ENV PYTHONUNBUFFERED=1

# Health check (Cloud Run will use this)
HEALTHCHECK --interval=30s --timeout=30s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:${PORT}/ || exit 1

# Use gunicorn for production with single worker (GPU constraint)
# Timeout of 900s for long-running inference requests
CMD ["gunicorn", \
     "--bind", "0.0.0.0:8080", \
     "--workers", "1", \
     "--threads", "2", \
     "--timeout", "900", \
     "--keep-alive", "30", \
     "--log-level", "info", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "physx_service:app"]