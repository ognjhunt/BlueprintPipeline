# syntax=docker/dockerfile:1
# physx-service/Dockerfile

FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

# --- System deps ----------------------------------------------------
# Minimal but enough to build common Python wheels (torch, CUDA extensions,
# PyAV/FFmpeg, etc.)
# Added libgl1 for OpenCV support.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 python3-venv python3-dev \
    git wget curl \
    build-essential pkg-config \
    ffmpeg \
    libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev libavutil-dev \
    libswresample-dev libswscale-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# --- Python venv ----------------------------------------------------
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Upgrade packaging tools
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# -------------------------------------------------------------------
# Install PyTorch FIRST so any CUDA extensions (CARAFE, CuVoxelization,
# chamferdist, etc.) that import torch in setup.py have it available.
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
#    IMPORTANT: we now let pip use its default build isolation so that
#    PEP 517 projects like pynim (PyNanoInstantMeshes) can automatically
#    pull their build backends (scikit-build-core, etc.).
RUN pip install --no-cache-dir -r requirements.txt

# 3a) Manually install chamferdist, which needs torch available during
#     its build, so we install it separately with build isolation off.
RUN pip install --no-cache-dir --no-build-isolation chamferdist==1.0.3

# 3b) Manually install CARAFE at the commit used by PhysX-Anything.
# We do this *after* torch is installed, and with build isolation
# disabled, because CARAFE's setup.py imports torch.
RUN pip install --no-cache-dir --no-build-isolation --no-deps \
    "carafe @ git+https://github.com/myownskyW7/CARAFE.git@aae89996131bcdb63d673c11db381bface05e751"

# 3c) Manually install CuVoxelization (CUDA voxelization ops) from GitHub.
RUN pip install --no-cache-dir --no-build-isolation --no-deps \
    "CuVoxelization @ git+https://github.com/lzhnb/CuVoxelization.git"

# 3d) Manually install EMD (Earth Mover's Distance).
# The requirements had 'emd==0.0.0' which is a local build.
# We install the standard CUDA implementation used in 3D research (MSN fork).
RUN pip install --no-cache-dir --no-build-isolation --no-deps \
    "git+https://github.com/Colin97/MSN-Point-Cloud-Completion.git#subdirectory=emd"

# 3e) Manually install Pointnet2 Ops.
# This often fails in requirements due to strict torch version matching.
RUN pip install --no-cache-dir --no-build-isolation --no-deps \
    "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

# 3f) Manually install Mip-Splatting's Gaussian rasterization extensions.
# PhysX-Anything's requirements reference local paths under
# tmp/extensions/mip-splatting/submodules/..., which do not exist inside
# Docker. We clone the official repo and install the two submodules that
# Trellis / Mip-Splatting normally use: diff-gaussian-rasterization and
# simple-knn.
RUN mkdir -p /tmp/extensions && \
    git clone --recurse-submodules https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting && \
    pip install --no-cache-dir --no-build-isolation --no-deps \
      /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization \
      /tmp/extensions/mip-splatting/submodules/simple-knn

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