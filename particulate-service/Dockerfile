# syntax=docker/dockerfile:1
# particulate-service/Dockerfile
#
# Particulate: Feed-Forward 3D Object Articulation
# Paper: arXiv:2512.11798
# Repo: https://github.com/RuiningLi/particulate
#
# CRITICAL CLOUD RUN SETTINGS (these are REQUIRED):
# ----------------------------------------------------------
# gcloud run deploy particulate-service \
#   --image gcr.io/PROJECT/particulate-service:v1 \
#   --memory 16Gi \
#   --cpu 4 \
#   --gpu 1 \
#   --gpu-type nvidia-l4 \
#   --timeout 300 \
#   --concurrency 1 \
#   --min-instances 0 \
#   --max-instances 4 \
#   --port 8080 \
#   --region us-central1 \
#   --no-cpu-throttling \
#   --set-env-vars "PARTICULATE_DEBUG=1"
# ----------------------------------------------------------
#
# IMPORTANT NOTES:
# - Fast inference: ~10s per object
# - Memory efficient: 16Gi
# - Model checkpoint auto-downloaded from HuggingFace (~100MB)
# - Fast cold starts: ~1-2 min
#

FROM nvidia/cuda:12.4.0-cudnn-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------
# System deps
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y \
    python3 python3-venv python3-dev python3-pip \
    git wget curl \
    build-essential cmake \
    libgl1 libglib2.0-0 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda

# ---------------------------------------------------------
# Python venv + tooling
# ---------------------------------------------------------
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ---------------------------------------------------------
# 1) Install PyTorch (GPU build) FIRST
# ---------------------------------------------------------
RUN pip install --no-cache-dir \
    torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Verify CUDA is available
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# ---------------------------------------------------------
# 2) Clone Particulate repository
# ---------------------------------------------------------
ENV PARTICULATE_ROOT=/opt/particulate
WORKDIR /opt

RUN git clone https://github.com/RuiningLi/particulate.git "${PARTICULATE_ROOT}"

WORKDIR ${PARTICULATE_ROOT}

# ---------------------------------------------------------
# 3) Install Particulate dependencies
# ---------------------------------------------------------

# Install from requirements.txt (with some fixes for version conflicts)
RUN pip install --no-cache-dir \
    lightning==2.2 \
    diffusers \
    h5py \
    yacs \
    omegaconf \
    trimesh \
    pymeshlab \
    plyfile \
    open3d \
    pygltflib \
    scikit-image \
    scipy \
    einops \
    psutil \
    loguru \
    simple_parsing \
    huggingface_hub

# Install mesh processing libraries (some may require special handling)
RUN pip install --no-cache-dir \
    potpourri3d \
    polyscope || true

# Install torch-scatter for the correct PyTorch version
RUN pip install --no-cache-dir torch-scatter \
    -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

# Try to install libigl (optional, may fail on some systems)
RUN pip install --no-cache-dir libigl || echo "libigl installation failed, continuing..."

# ---------------------------------------------------------
# 4) Clone PartField (required dependency)
# ---------------------------------------------------------
# PartField is used for feature extraction
RUN if [ ! -d "PartField" ]; then \
        git clone https://github.com/SamsungLabs/PartField.git PartField; \
    fi

# ---------------------------------------------------------
# 5) Download model checkpoint (warm the cache)
# ---------------------------------------------------------
RUN python -c "
from huggingface_hub import hf_hub_download
checkpoint = hf_hub_download(repo_id='rayli/Particulate', filename='model.pt')
print(f'Model downloaded to: {checkpoint}')
"

# Also download PartField features extractor if needed
RUN python -c "
from huggingface_hub import hf_hub_download
try:
    # PartField checkpoint (if available on HF)
    hf_hub_download(repo_id='rayli/Particulate', filename='partfield.pt')
except:
    print('PartField checkpoint not on HF, will use local')
" || true

# ---------------------------------------------------------
# 6) Create inference wrapper script
# ---------------------------------------------------------
COPY particulate_infer_wrapper.py ${PARTICULATE_ROOT}/particulate_infer_wrapper.py

# ---------------------------------------------------------
# 7) Install Flask and service dependencies
# ---------------------------------------------------------
RUN pip install --no-cache-dir \
    flask \
    gunicorn

# ---------------------------------------------------------
# 8) Copy service files
# ---------------------------------------------------------
WORKDIR /app

COPY particulate_service.py /app/particulate_service.py

# Create temp directory for request processing
RUN mkdir -p /tmp/particulate && chmod 777 /tmp/particulate

# ---------------------------------------------------------
# Environment and startup configuration
# ---------------------------------------------------------
ENV PORT=8080
ENV PARTICULATE_ROOT=/opt/particulate
ENV PARTICULATE_DEBUG=0
ENV PARTICULATE_HEALTHCHECK_HOST=localhost
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://${PARTICULATE_HEALTHCHECK_HOST}:${PORT}/ || exit 1

# Use gunicorn for production
CMD ["gunicorn", \
     "--bind", "0.0.0.0:8080", \
     "--workers", "1", \
     "--threads", "2", \
     "--timeout", "300", \
     "--graceful-timeout", "60", \
     "--keep-alive", "30", \
     "--log-level", "info", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--capture-output", \
     "particulate_service:app"]
