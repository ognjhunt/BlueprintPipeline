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
# 1) Install the sam3 package from PyPI to pull in its declared deps.
#    (Package itself has some layout issues, but we use the source tree
#    for code via PYTHONPATH, and keep this for dependency resolution.)
RUN pip install --no-cache-dir --prefer-binary \
        sam3 \
        opencv-python-headless \
        Pillow \
        numpy \
        PyYAML \
        decord \
        psutil

# 2) Clone the official SAM3 repo and make it importable directly.
#    This works around the 'No module named sam3.sam' packaging bug by
#    letting Python import from the source tree instead of the incomplete wheel.
RUN git clone https://github.com/facebookresearch/sam3.git /opt/sam3

# ---------------------------
# Runtime environment
# ---------------------------
ENV HF_HUB_DISABLE_TELEMETRY=1 \
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
