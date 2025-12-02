FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# System dependencies:
# - python3 / pip / git / certs
# - libglib2.0-0 provides libgthread-2.0.so.0 (needed by OpenCV)
# - libgl1, libsm6, libxext6, libxrender1 are common OpenCV runtime deps
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 python3-pip git ca-certificates \
    libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Make "python" point to python3 and upgrade pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    python -m pip install --upgrade pip

# Clone Depth Anything 3 repo
WORKDIR /app
RUN git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git /app/Depth-Anything-3

# Install PyTorch (cu121) + huggingface_hub (pinned) + Depth Anything 3 + GCS client
WORKDIR /app/Depth-Anything-3
RUN pip install --no-cache-dir --prefer-binary \
      torch==2.2.1 torchvision==0.17.1 \
      --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir --prefer-binary "huggingface_hub==0.33.5" && \
    pip install --no-cache-dir --prefer-binary "google-cloud-storage>=2.0.0" && \
    pip install --no-cache-dir --prefer-binary -e .

# Hugging Face env:
# - Disable Xet-based downloads (to avoid xet-read-token 429s)
# - Turn off telemetry
# - Cache under /mnt/gcs so model files persist across job runs
ENV HF_HUB_DISABLE_XET=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HOME=/mnt/gcs/hf-cache \
    TORCH_HOME=/tmp/torch-cache \
    PYTHONUNBUFFERED=1

# Back to /app for our job code
WORKDIR /app

# Job scripts
COPY run_da3_scene_from_dataset.py /app/run_da3_scene_from_dataset.py
COPY run_da3_scene.sh /app/run_da3_scene.sh

RUN chmod +x /app/run_da3_scene.sh

ENTRYPOINT ["/app/run_da3_scene.sh"]
