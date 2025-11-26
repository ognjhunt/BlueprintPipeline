# syntax=docker/dockerfile:1
# physx-service/Dockerfile

FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

# ---------------------------------------------------------
# System deps (enough to build the usual CUDA/PyTorch wheels)
# ---------------------------------------------------------
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 python3-venv python3-dev \
    git wget curl \
    build-essential pkg-config \
    ffmpeg \
    libgl1 \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# Python venv + tooling
# ---------------------------------------------------------
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ---------------------------------------------------------
# 1) Install PyTorch *first*
#    We use a combo known to work well with TRELLIS in
#    containerized setups: torch 2.5.0 + cu124 wheels.
# ---------------------------------------------------------
RUN pip install --no-cache-dir \
    torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
    --index-url https://download.pytorch.org/whl/cu124

# ---------------------------------------------------------
# 2) Clone PhysX-Anything (with submodules: TRELLIS, etc.)
# ---------------------------------------------------------
ENV PHYSX_ROOT=/opt/physx_anything
WORKDIR /opt

RUN git clone --recurse-submodules \
    https://github.com/ziangcao0312/PhysX-Anything.git "${PHYSX_ROOT}"

WORKDIR ${PHYSX_ROOT}

# ---------------------------------------------------------
# 3) Heavy CUDA stack: Kaolin + TRELLIS setup.sh
# ---------------------------------------------------------
# Kaolin 0.15.0 from NVIDIA's wheel index for torch-2.5.0/cu121.
# This fixes the "No matching distribution found for kaolin==0.15.0"
# error you hit when relying on plain PyPI.
RUN pip install --no-cache-dir \
    "kaolin==0.15.0" \
    -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.0_cu121.html

# Use TRELLIS / PhysX setup.sh to install:
#   - xformers
#   - flash-attn
#   - diffoctreerast
#   - spconv (+ cumm, pccm)
#   - mip-splatting (diff-gaussian-rasterization + simple-knn)
#   - kaolin bits
#   - nvdiffrast
# We *do not* pass --new-env so it installs into the current venv.
RUN bash -lc ". ./setup.sh --basic --xformers --flash-attn --diffoctreerast \
                         --spconv --mipgaussian --kaolin --nvdiffrast"

# ---------------------------------------------------------
# 4) Light-weight Python deps from patched requirements
#    (strip cluster-local wheels + heavy CUDA extensions)
# ---------------------------------------------------------
COPY patch_requirements.py /tmp/patch_requirements.py

# Write a cleaned-up requirements file alongside the original.
RUN python /tmp/patch_requirements.py requirements.txt requirements.patched.txt

# Install only the "safe" / generic deps (numpy, pandas, mmcv, etc.)
RUN pip install --no-cache-dir -r requirements.patched.txt

# ---------------------------------------------------------
# 5) Extra CUDA ops that TRELLIS doesn't manage
# ---------------------------------------------------------
# chamferdist needs torch available during its build, so we
# disable build isolation.
RUN pip install --no-cache-dir --no-build-isolation chamferdist==1.0.3

# CARAFE
RUN pip install --no-cache-dir --no-build-isolation --no-deps \
    "carafe @ git+https://github.com/myownskyW7/CARAFE.git@aae89996131bcdb63d673c11db381bface05e751"

# CuVoxelization
RUN pip install --no-cache-dir --no-build-isolation --no-deps \
    "CuVoxelization @ git+https://github.com/lzhnb/CuVoxelization.git"

# EMD from MSN repo
RUN pip install --no-cache-dir --no-build-isolation --no-deps \
    "git+https://github.com/Colin97/MSN-Point-Cloud-Completion.git#subdirectory=emd"

# PointNet2 ops
RUN pip install --no-cache-dir --no-build-isolation --no-deps \
    "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

# ---------------------------------------------------------
# 6) Download PhysX-Anything/TRELLIS checkpoints
# ---------------------------------------------------------
RUN python download.py

# ---------------------------------------------------------
# 7) Your Flask wrapper service
# ---------------------------------------------------------
WORKDIR /app

RUN pip install --no-cache-dir flask

COPY run_physx_anything_pipeline.py ${PHYSX_ROOT}/run_physx_anything_pipeline.py
COPY physx_service.py /app/physx_service.py

ENV PORT=8080

CMD ["python", "physx_service.py"]
