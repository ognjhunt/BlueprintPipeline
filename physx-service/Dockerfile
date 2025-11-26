# syntax=docker/dockerfile:1
# physx-service/Dockerfile

FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

# ---------------------------------------------------------
# System deps
# ---------------------------------------------------------
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 python3-venv python3-dev \
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
#    Torch 2.5.0 + cu124, per official index. :contentReference[oaicite:5]{index=5}
# ---------------------------------------------------------
RUN pip install --no-cache-dir \
    torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
    --index-url https://download.pytorch.org/whl/cu124

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
#    (0.15.0 does NOT exist for this combo; use 0.17.0). :contentReference[oaicite:6]{index=6}
# ---------------------------------------------------------
RUN pip install --no-cache-dir \
    "kaolin==0.17.0" \
    -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.0_cu124.html

# ---------------------------------------------------------
# 4) Patch requirements.txt IN PLACE so any internal
#    'pip install -r requirements.txt' sees a sane file
# ---------------------------------------------------------
COPY patch_requirements.py /tmp/patch_requirements.py
RUN python /tmp/patch_requirements.py requirements.txt

# ---------------------------------------------------------
# 5) Use setup.sh to install the heavy CUDA stack
#    as recommended by the PhysX-Anything/TRELLIS docs. :contentReference[oaicite:7]{index=7}
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
# 6) Qwen2.5 deps (as in PhysX-Anything README) :contentReference[oaicite:8]{index=8}
# ---------------------------------------------------------
RUN pip install --no-cache-dir \
      'transformers==4.50.0' \
      qwen-vl-utils \
      'accelerate>=0.26.0'

# Optionally download model weights (if the repo provides this script).
RUN if [ -f download.py ]; then python download.py; fi

# ---------------------------------------------------------
# 7) Your Flask wrapper service
# ---------------------------------------------------------
WORKDIR /app
RUN pip install --no-cache-dir flask

COPY run_physx_anything_pipeline.py ${PHYSX_ROOT}/run_physx_anything_pipeline.py
COPY physx_service.py /app/physx_service.py

ENV PORT=8080
CMD ["python", "physx_service.py"]
