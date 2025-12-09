# interactive-job/Dockerfile
#
# Interactive Asset Pipeline for ZeroScene Integration
#
# This job processes 3D assets from ZeroScene to add articulation data
# using PhysX-Anything.
#
# CLOUD RUN JOB SETTINGS:
# ----------------------------------------------------------
# gcloud run jobs create interactive-job \
#   --image gcr.io/PROJECT/interactive-job:latest \
#   --memory 4Gi \
#   --cpu 2 \
#   --timeout 3600 \
#   --max-retries 1 \
#   --region us-central1
# ----------------------------------------------------------

FROM python:3.11-slim

# Install system dependencies for trimesh rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
# - trimesh: Mesh loading and rendering
# - numpy: Numerical operations
# - Pillow: Image processing for rendering
# - pyglet: Backend for trimesh rendering (optional, helps with offscreen rendering)
RUN pip install --no-cache-dir \
    trimesh[easy] \
    numpy \
    Pillow \
    pyglet

# Copy application files
COPY run_interactive_assets.py /app/run_interactive_assets.py

# Create temp directory for mesh rendering
RUN mkdir -p /tmp/interactive && chmod 777 /tmp/interactive

# Environment configuration
ENV PYTHONUNBUFFERED=1
ENV INTERACTIVE_MODE=glb

# Default command
CMD ["python", "run_interactive_assets.py"]
