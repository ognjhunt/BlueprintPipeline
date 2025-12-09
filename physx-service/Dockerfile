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
# System deps (including git-lfs for model downloads)
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y \
    python3 python3-venv python3-dev python3-pip \
    git git-lfs wget curl \
    build-essential cmake ninja-build pkg-config \
    ffmpeg \
    libgl1 libglib2.0-0 \
    && git lfs install \
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

# Accept HF_TOKEN as build argument for authentication
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Copy our robust download script
COPY download_models.py /tmp/download_models.py

# Download models with authentication and retry logic
RUN echo "========================================" && \
    echo "Downloading PhysX-Anything models..." && \
    echo "========================================" && \
    python /tmp/download_models.py \
        --repo-id "Caoza/PhysX-Anything" \
        --output-dir "${PHYSX_ROOT}" \
        --token "${HF_TOKEN}" && \
    echo "========================================" && \
    echo "Model download completed successfully!" && \
    echo "========================================" || \
    (echo "ERROR: Model download failed!" && exit 1)

# Unset token for security
ENV HF_TOKEN=""

# Final verification with detailed output
RUN echo "========================================" && \
    echo "Final model verification..." && \
    echo "========================================" && \
    if [ -d "pretrain/vlm" ]; then \
        echo "✓ VLM directory exists" && \
        echo "" && \
        echo "Files present:" && \
        ls -lh pretrain/vlm/ && \
        echo "" && \
        echo "Total size:" && \
        du -sh pretrain/vlm/ && \
        echo "" && \
        echo "Checking required files..." && \
        python -c "
import sys
from pathlib import Path

vlm_dir = Path('pretrain/vlm')
files = [f.name for f in vlm_dir.glob('*') if f.is_file()]

required = ['config.json', 'tokenizer.json', 'tokenizer_config.json']
missing = [f for f in required if f not in files]

has_weights = any(f.endswith('.safetensors') or f.endswith('.bin') for f in files)

if missing:
    print(f'ERROR: Missing files: {missing}')
    sys.exit(1)

if not has_weights:
    print('ERROR: No model weights found')
    sys.exit(1)

print('✓ All required files present')
print(f'✓ Model weights found ({\"safetensors\" if any(f.endswith(\".safetensors\") for f in files) else \"bin\"})')
" && \
        echo "========================================" && \
        echo "✓ Model verification PASSED" && \
        echo "========================================" ; \
    else \
        echo "ERROR: VLM directory not found!" && \
        exit 1 ; \
    fi

# ---------------------------------------------------------
# 8) Install additional dependencies
# ---------------------------------------------------------
RUN pip install --no-cache-dir \
    ipdb \
    flask \
    gunicorn \
    trimesh \
    Pillow \
    pyglet

# ---------------------------------------------------------
# 9) Copy service files
# ---------------------------------------------------------
WORKDIR /app

# Copy the pipeline wrapper
COPY run_physx_anything_pipeline.py ${PHYSX_ROOT}/run_physx_anything_pipeline.py

# Copy mesh rendering utility (for GLB → image conversion)
COPY mesh_to_views.py /app/mesh_to_views.py

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