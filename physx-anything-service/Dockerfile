# syntax=docker/dockerfile:1
# physx-anything-service/Dockerfile
#
# PhysX-Anything: Turning a Single Image into a Physics Simulation Ready World
# Repo: https://github.com/VictorTao1998/PhysX-Anything
#
# This is a GPU container that exposes a small HTTP API returning a zip payload
# (URDF + referenced meshes) for use by interactive-job.

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-venv python3-dev python3-pip \
    git wget curl \
    build-essential cmake \
    libgl1 libglib2.0-0 libxrender1 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install GPU PyTorch first (avoid pulling CPU wheels from the upstream requirements file).
RUN pip install --no-cache-dir \
    torch==2.3.1 torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cu121

ENV PHYSX_ANYTHING_ROOT=/opt/physx-anything
WORKDIR /opt
RUN git clone https://github.com/VictorTao1998/PhysX-Anything.git "${PHYSX_ANYTHING_ROOT}"

WORKDIR ${PHYSX_ANYTHING_ROOT}

# Install requirements but omit torch/torchvision (already installed above).
RUN python -c "import pathlib; p=pathlib.Path('requirements.txt'); txt=p.read_text().splitlines(); txt=[l for l in txt if not l.strip().startswith('torch==') and not l.strip().startswith('torchvision==')]; pathlib.Path('/tmp/requirements.txt').write_text('\\n'.join(txt)+'\\n')"
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# download.py depends on gdown.
RUN pip install --no-cache-dir gdown

# Pre-download all model artifacts at build time (no runtime downloads).
RUN python download.py

RUN pip install --no-cache-dir flask gunicorn

WORKDIR /app
COPY physx_anything_service.py /app/physx_anything_service.py

ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV PHYSX_ANYTHING_ROOT=/opt/physx-anything
ENV PHYSX_ANYTHING_CKPT=/opt/physx-anything/pretrain/vlm

HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${PORT}/ || exit 1

CMD ["gunicorn", \
     "--bind", "0.0.0.0:8080", \
     "--workers", "1", \
     "--threads", "2", \
     "--timeout", "1200", \
     "--graceful-timeout", "60", \
     "--keep-alive", "30", \
     "--log-level", "info", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--capture-output", \
     "physx_anything_service:app"]

