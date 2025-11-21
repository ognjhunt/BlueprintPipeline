FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

# ---------------------------
# System dependencies
# ---------------------------
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    software-properties-common \
    git \
    ca-certificates \
    ffmpeg \
    libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------
# Python 3.12
# ---------------------------
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    python -m ensurepip --upgrade

# ---------------------------
# PyTorch (match SAM3 docs: torch 2.7.0 + CUDA 12.6)
# ---------------------------
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --prefer-binary \
        torch==2.7.0 torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu126

# ---------------------------
# SAM3 + supporting Python deps
# ---------------------------
# Core scientific + HF stack. Note: we do NOT install transformers here.
RUN pip install --no-cache-dir --prefer-binary \
        sam3 \
        opencv-python-headless \
        Pillow \
        numpy \
        PyYAML \
        decord \
        psutil \
        huggingface_hub \
        accelerate \
        safetensors

# Install Transformers from GitHub main so Sam3Model / Sam3Processor are available.
# Docs: the SAM3 page is on "main" and explicitly says to install from source.:contentReference[oaicite:2]{index=2}
RUN pip install --no-cache-dir --prefer-binary \
        "git+https://github.com/huggingface/transformers.git@main"

# ---------------------------
# (Optional) SAM3 research repo clone
# ---------------------------
RUN git clone https://github.com/facebookresearch/sam3.git /opt/sam3

# ---------------------------
# Runtime environment
# ---------------------------
ENV HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_XET=1 \
    HUGGINGFACE_HUB_DISABLE_XET=1 \
    HF_HUB_ENABLE_XET=0 \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_HOME=/mnt/gcs/hf-cache \
    TRANSFORMERS_CACHE=/mnt/gcs/hf-cache \
    TORCH_HOME=/tmp/torch-cache \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/opt/sam3:${PYTHONPATH}"

WORKDIR /app

# Job scripts
COPY run_sam3_from_images.py /app/run_sam3_from_images.py
COPY run_sam3.sh /app/run_sam3.sh

RUN chmod +x /app/run_sam3.sh

ENTRYPOINT ["/app/run_sam3.sh"]
