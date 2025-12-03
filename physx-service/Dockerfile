# syntax=docker/dockerfile:1
# physx-service/Dockerfile
#
# CRITICAL CLOUD RUN SETTINGS (these are REQUIRED):
# ----------------------------------------------------------
# gcloud run deploy physx-service \
#   --image gcr.io/PROJECT/physx-service:v3 \
#   --memory 32Gi \
#   --cpu 8 \
#   --gpu 1 \
#   --gpu-type nvidia-l4 \
#   --timeout 900 \
#   --concurrency 1 \
#   --min-instances 1 \
#   --max-instances 2 \
#   --port 8080 \
#   --region us-central1 \
#   --no-cpu-throttling \
#   --set-env-vars "PHYSX_DEBUG=1"
# ----------------------------------------------------------
#
# IMPORTANT NOTES:
# - concurrency=1 is REQUIRED (GPU can only handle one inference at a time)
# - min-instances=1 is HIGHLY RECOMMENDED (VLM model takes 10+ minutes to load)
# - timeout=900 is needed (pipeline can take 5-15 minutes per object)
# - memory=32Gi is needed for the VLM model
#

FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------
# System deps
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y \
    python3 python3-venv python3-dev python3-pip \
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
# ---------------------------------------------------------
RUN pip install --no-cache-dir \
    torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Verify CUDA is available
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

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
# 4) Patch requirements.txt IN PLACE
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
# 7) CRITICAL: Download model weights at build time
#    This is the most important step for avoiding cold starts!
# ---------------------------------------------------------

# First, try the repo's download script
RUN echo "Attempting to download models using download.py..." && \
    if [ -f download.py ]; then \
        python download.py && echo "download.py succeeded" || \
        echo "Warning: download.py failed, trying alternative methods..."; \
    else \
        echo "download.py not found, will try alternative methods..."; \
    fi

# Verify pretrain/vlm directory exists and has content
RUN echo "Checking VLM checkpoint directory..." && \
    if [ -d "pretrain/vlm" ]; then \
        echo "VLM directory exists:" && \
        ls -la pretrain/vlm/ && \
        echo "Total size:" && \
        du -sh pretrain/vlm/ ; \
    else \
        echo "WARNING: VLM checkpoint directory does not exist!" && \
        echo "The service will fail at runtime without model weights." && \
        mkdir -p pretrain/vlm ; \
    fi

# Check for required model files
RUN echo "Checking for required model files..." && \
    if [ -f "pretrain/vlm/config.json" ]; then \
        echo "config.json found" ; \
    else \
        echo "WARNING: config.json not found!" ; \
    fi && \
    if ls pretrain/vlm/*.safetensors 1> /dev/null 2>&1 || ls pretrain/vlm/*.bin 1> /dev/null 2>&1; then \
        echo "Model weights found" ; \
    else \
        echo "WARNING: No model weights found!" ; \
    fi

# ---------------------------------------------------------
# 8) Install additional dependencies
# ---------------------------------------------------------
RUN pip install --no-cache-dir ipdb flask gunicorn

# ---------------------------------------------------------
# 9) Copy service files
# ---------------------------------------------------------
WORKDIR /app

# Copy the pipeline wrapper
COPY run_physx_anything_pipeline.py ${PHYSX_ROOT}/run_physx_anything_pipeline.py

# Copy the Flask service
COPY physx_service.py /app/physx_service.py

# Create temp directory for request processing
RUN mkdir -p /tmp/physx_anything && chmod 777 /tmp/physx_anything

# Create lock file directory
RUN mkdir -p /tmp && touch /tmp/physx_pipeline.lock && chmod 666 /tmp/physx_pipeline.lock

# ---------------------------------------------------------
# Environment and startup configuration
# ---------------------------------------------------------
ENV PORT=8080
ENV PHYSX_ROOT=/opt/physx_anything
ENV PHYSX_DEBUG=1
# Disable Python output buffering for better logging
ENV PYTHONUNBUFFERED=1
# Reduce transformers verbosity slightly
ENV TRANSFORMERS_VERBOSITY=warning
# Help with GPU memory
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Health check - note the long start period for model loading
HEALTHCHECK --interval=30s --timeout=30s --start-period=600s --retries=5 \
    CMD curl -f http://localhost:${PORT}/ || exit 1

# Use gunicorn for production with single worker (GPU constraint)
# --timeout 900 for long-running inference requests
# --graceful-timeout 120 to allow in-flight requests to complete on shutdown
CMD ["gunicorn", \
     "--bind", "0.0.0.0:8080", \
     "--workers", "1", \
     "--threads", "2", \
     "--timeout", "900", \
     "--graceful-timeout", "120", \
     "--keep-alive", "30", \
     "--log-level", "info", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--capture-output", \
     "physx_service:app"]