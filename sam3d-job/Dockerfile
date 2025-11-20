FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

# System deps for Python + vision libs
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    software-properties-common \
    git \
    ca-certificates \
    libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3.12-distutils \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    python -m ensurepip --upgrade

RUN python -m pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefer-binary \
    torch==2.7.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

RUN pip install --no-cache-dir --prefer-binary \
    huggingface_hub \
    git+https://github.com/facebookresearch/sam-3d-objects.git \
    opencv-python-headless \
    Pillow \
    numpy \
    PyYAML \
    usd-core==24.05

ENV SAM3D_REPO_ROOT=/app/sam3d-objects \
    SAM3D_CHECKPOINT_ROOT=/app/sam3d-assets/checkpoints

# Keep a local checkout for configs/notebooks
RUN git clone https://github.com/facebookresearch/sam-3d-objects.git ${SAM3D_REPO_ROOT} || true
RUN if [ -f ${SAM3D_REPO_ROOT}/requirements.inference.txt ]; then \
      pip install --no-cache-dir -r ${SAM3D_REPO_ROOT}/requirements.inference.txt; \
    fi
RUN mkdir -p ${SAM3D_CHECKPOINT_ROOT}

ENV HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HOME=/mnt/gcs/hf-cache \
    TRANSFORMERS_CACHE=/mnt/gcs/hf-cache \
    TORCH_HOME=/tmp/torch-cache \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY run_sam3d_from_assets.py /app/run_sam3d_from_assets.py
COPY run_sam3d.sh /app/run_sam3d.sh

RUN chmod +x /app/run_sam3d.sh

ENTRYPOINT ["/app/run_sam3d.sh"]
