# interactive-job/Dockerfile
#
# Interactive Asset Pipeline (Multi-Backend) for BlueprintPipeline.
#
# This image runs the interactive-job entrypoint:
#   python interactive-job/run_interactive_assets.py
#
# It is intentionally lightweight and can be paired with GPU backend services:
# - particulate-service (mesh -> URDF)
# - physx-anything-service (image -> URDF)
# - (optional) infinigen-service (category -> URDF)
#
# Build from repo root:
#   docker build -f interactive-job/Dockerfile -t blueprint-interactive:latest .

FROM python:3.11-slim

# System deps for trimesh offscreen rendering + pybullet
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY interactive-job/requirements.txt ./interactive-job/requirements.txt
RUN pip install --no-cache-dir -r interactive-job/requirements.txt

# Copy job + shared tools (keep image self-contained; no runtime git pulls)
COPY interactive-job/ ./interactive-job/
COPY tools/ ./tools/
COPY monitoring/ ./monitoring/

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV INTERACTIVE_MODE=glb

RUN mkdir -p /tmp/interactive && chmod 777 /tmp/interactive

CMD ["python", "interactive-job/run_interactive_assets.py"]

