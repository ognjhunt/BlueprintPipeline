FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Basic system deps
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 python3-pip git ca-certificates \
    libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Make "python" point to python3 and upgrade pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    python -m pip install --upgrade pip

# Install PyTorch with CUDA 12.1 wheels (GPU-enabled)
RUN pip install --no-cache-dir --prefer-binary \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Pin huggingface_hub to a version where HF_HUB_DISABLE_XET is effective
RUN pip install --no-cache-dir --prefer-binary \
    "huggingface_hub==0.33.5"

# Install Autodistill Grounded SAM + roboflow + scikit-learn + utilities
# NOTE: We use autodistill-grounded-sam (SAM1) to avoid flash-attn issues
RUN pip install --no-cache-dir --prefer-binary \
    autodistill \
    autodistill-grounded-sam \
    roboflow \
    scikit-learn \
    supervision \
    opencv-python-headless \
    Pillow

# HF / Torch caches under /mnt/gcs so they persist across runs
# HF_HUB_DISABLE_XET avoids the xet-read-token path that is causing 429s
ENV HF_HUB_DISABLE_XET=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HOME=/mnt/gcs/hf-cache \
    TRANSFORMERS_CACHE=/mnt/gcs/hf-cache \
    TORCH_HOME=/tmp/torch-cache \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Job scripts
COPY run_gsam2_from_images.py /app/run_gsam2_from_images.py
COPY run_gsam2.sh /app/run_gsam2.sh

RUN chmod +x /app/run_gsam2.sh

ENTRYPOINT ["/app/run_gsam2.sh"]
