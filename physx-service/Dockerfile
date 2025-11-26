# syntax=docker/dockerfile:1
# physx-service/Dockerfile

FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

# --- System deps ----------------------------------------------------
# Minimal but enough to build common Python wheels (torch, etc.)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 python3-venv python3-dev \
    git wget curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- Python venv ----------------------------------------------------
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Upgrade packaging tools
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# -------------------------------------------------------------------
# Install PyTorch FIRST so any CUDA extensions (CARAFE, mmcv ops, etc.)
# that import torch in setup.py have it available.:contentReference[oaicite:3]{index=3}
# -------------------------------------------------------------------
RUN pip install --no-cache-dir torch torchvision torchaudio

# -------------------------------------------------------------------
# Install PhysX-Anything into /opt/physx_anything
# -------------------------------------------------------------------
ENV PHYSX_ROOT=/opt/physx_anything

WORKDIR /opt

# 1) Clone the official PhysX-Anything repo
RUN git clone https://github.com/ziangcao0312/PhysX-Anything.git "${PHYSX_ROOT}"

WORKDIR ${PHYSX_ROOT}

# 2) Copy and run the requirements patcher
COPY patch_requirements.py /tmp/patch_requirements.py
RUN python /tmp/patch_requirements.py requirements.txt

# 3) Install Python dependencies from the (patched) requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 3b) Manually install CARAFE at the commit used by PhysX-Anything.
# We do this *after* torch is installed, and with build isolation
# disabled, because CARAFE's setup.py imports torch.:contentReference[oaicite:4]{index=4}
RUN pip install --no-cache-dir --no-build-isolation --no-deps \
    "carafe @ git+https://github.com/myownskyW7/CARAFE.git@aae89996131bcdb63d673c11db381bface05e751"

# 4) Download pretrained weights
RUN python download.py

# -------------------------------------------------------------------
# Our wrapper service + small pipeline driver
# -------------------------------------------------------------------
WORKDIR /app

# Flask for the HTTP service
RUN pip install --no-cache-dir flask

# Copy the driver script into the PhysX-Anything repo
COPY run_physx_anything_pipeline.py ${PHYSX_ROOT}/run_physx_anything_pipeline.py

# Copy the HTTP wrapper
COPY physx_service.py /app/physx_service.py

# Cloud Run expects the server to listen on this port
ENV PORT=8080

# Start the Flask app
CMD ["python", "physx_service.py"]
