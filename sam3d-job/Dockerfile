FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

# System deps for Python + vision libs + building CUDA extensions
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    software-properties-common \
    git \
    ca-certificates \
    build-essential \
    libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.12 and venv tooling
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      python3.12 \
      python3.12-venv \
      python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

# Make python point to 3.12, bootstrap pip, and create a dedicated venv
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    python -m ensurepip --upgrade && \
    python -m venv /opt/venv

# From here on, always use the venv's Python/pip
ENV PATH="/opt/venv/bin:${PATH}"

# Upgrade pip/setuptools/wheel inside the venv
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch (CUDA 12.4 build) in the venv
RUN pip install --no-cache-dir --prefer-binary \
    torch==2.5.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124
# Using stable torch 2.5.1+cu124, compatible with CUDA 12.x.

# -------------------------------------------------------------------
# Core runtime deps (WITHOUT sam-3d-objects yet)
# Note: usd-core valid versions start at 24.8
# We ALSO install hydra-core + omegaconf so notebook/inference.py can import them.
#
# IMPORTANT: we DO need the PyPI "utils3d" package because notebook/inference.py
# does `import utils3d`, but that package depends on open3d and pip's resolver
# will fail if it tries to satisfy that dependency in this environment.
#
# To avoid the build failure while still satisfying the import, we:
#   - install our base deps normally
#   - then install utils3d with `--no-deps` so pip does NOT try to install open3d
# -------------------------------------------------------------------
RUN pip install --no-cache-dir --prefer-binary \
      huggingface_hub \
      opencv-python-headless \
      Pillow \
      numpy \
      PyYAML \
      "usd-core>=24.8" \
      hydra-core \
      omegaconf \
    && pip install --no-cache-dir --prefer-binary --no-deps \
      "utils3d==0.1.3"

# Install sam-3d-objects *without* pulling in its dependencies such as auto-gptq.
# We manage important deps ourselves in this venv.
RUN pip install --no-cache-dir --prefer-binary --no-deps \
    git+https://github.com/facebookresearch/sam-3d-objects.git

ENV SAM3D_REPO_ROOT=/app/sam3d-objects \
    SAM3D_CHECKPOINT_ROOT=/app/sam3d-assets/checkpoints

# Keep a local checkout for configs/notebooks (separate from the pip-installed package)
RUN git clone https://github.com/facebookresearch/sam-3d-objects.git ${SAM3D_REPO_ROOT} || true

# Work around PyTorch cpp_extension bug when building CUDA extensions (gsplat, kaolin, etc.)
# in an environment with no visible GPU during docker build.
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"

# Install inference requirements, but strip out auto-gptq (not needed for inference and
# expensive to build). Use --no-build-isolation so packages like gsplat can see torch
# and CUDA from the venv when building their CUDA extensions.
# IMPORTANT: Kaolin wheels (kaolin==0.17.0) are not on PyPI; we must point pip at
# NVIDIA's wheel index for torch 2.5.1 + cu124 so that requirement can be satisfied.
RUN if [ -f "${SAM3D_REPO_ROOT}/requirements.inference.txt" ]; then \
      grep -vi 'auto-gptq' "${SAM3D_REPO_ROOT}/requirements.inference.txt" > /tmp/requirements.inference.no-autogptq.txt; \
      pip install --no-cache-dir --prefer-binary --no-build-isolation \
        -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html \
        -r /tmp/requirements.inference.no-autogptq.txt; \
    fi

RUN mkdir -p ${SAM3D_CHECKPOINT_ROOT}

# Runtime env defaults
ENV HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HOME=/mnt/gcs/hf-cache \
    TRANSFORMERS_CACHE=/mnt/gcs/hf-cache \
    TORCH_HOME=/tmp/torch-cache \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    CONDA_PREFIX=/usr/local/cuda

WORKDIR /app

# Job entry scripts
COPY run_sam3d_from_assets.py /app/run_sam3d_from_assets.py
COPY run_sam3d.sh /app/run_sam3d.sh

RUN chmod +x /app/run_sam3d.sh

ENTRYPOINT ["/app/run_sam3d.sh"]
