FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

# --- System deps ----------------------------------------------------
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 python3-venv python3-dev \
    git wget curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- Python venv ----------------------------------------------------
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

# --- Install PhysX-Anything ----------------------------------------
# TODO: replace with the *actual* repo URL and setup commands.
# The pattern is:
#   1) clone repo
#   2) install its Python deps
#   3) download weights
#
# Example skeleton (YOU must edit):
#
# RUN git clone https://github.com/YOUR_ORG/PhysX-Anything.git /opt/physx_anything
# WORKDIR /opt/physx_anything
# RUN pip install --no-cache-dir -r requirements.txt
# RUN python download_weights.py
#
# After you’ve added the real commands, leave WORKDIR back at /app.
ENV PHYSX_ROOT=/opt/physx_anything
WORKDIR /app

# --- Install service deps (Flask etc.) ------------------------------
RUN pip install --no-cache-dir flask

# If PhysX-Anything needs extra libs like numpy, torch, etc.,
# that should already be installed by its own setup commands above.

# --- Copy our wrapper service --------------------------------------
COPY physx_service.py /app/physx_service.py

# Cloud Run expects the server to listen on this port
ENV PORT=8080

CMD ["python", "physx_service.py"]
