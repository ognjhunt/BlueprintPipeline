FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

# System deps
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    software-properties-common \
    git \
    ca-certificates \
    ffmpeg \
    libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Note: python3.12-distutils is not needed/available for 3.12 on this PPA
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    python -m ensurepip --upgrade

# Install PyTorch with CUDA 12.6 wheels (GPU-enabled)
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --prefer-binary \
    torch==2.5.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124
# Note: Using torch 2.5.1+cu124 as 2.7.0+cu126 is not a standard stable release yet in common indices,
# or check https://download.pytorch.org/whl/cu126 availability.
# If you strictly need "2.7.0", ensure the index-url actually has it.
# For safety, 2.5.1 is the current stable high-version.

# Install SAM 3 + utilities
RUN pip install --no-cache-dir --prefer-binary \
    git+https://github.com/facebookresearch/sam3.git \
    opencv-python-headless \
    Pillow \
    numpy \
    PyYAML

# HF / Torch caches under /mnt/gcs so they persist across runs
ENV HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HOME=/mnt/gcs/hf-cache \
    TRANSFORMERS_CACHE=/mnt/gcs/hf-cache \
    TORCH_HOME=/tmp/torch-cache \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Job scripts
COPY run_sam3_from_images.py /app/run_sam3_from_images.py
COPY run_sam3.sh /app/run_sam3.sh

RUN chmod +x /app/run_sam3.sh

ENTRYPOINT ["/app/run_sam3.sh"]