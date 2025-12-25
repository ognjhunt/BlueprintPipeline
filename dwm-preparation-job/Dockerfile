# Dockerfile for DWM Preparation Job
# Generates DWM conditioning bundles from BlueprintPipeline scenes
#
# DWM (Dexterous World Model) Reference: https://snuvclab.github.io/dwm/

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for optional MANO models
RUN mkdir -p /mano_models

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV MANO_MODEL_PATH=/mano_models

# Entry point - uses environment variables for configuration
# Required: BUCKET, SCENE_ID
# Optional: ASSETS_PREFIX, USD_PREFIX, DWM_PREFIX, NUM_TRAJECTORIES, etc.
ENTRYPOINT ["python", "entrypoint.py"]
