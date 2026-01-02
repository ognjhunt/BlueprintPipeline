# Dream2Flow Preparation Job Dockerfile
# Based on Dream2Flow (arXiv:2512.24766)
#
# Generates Dream2Flow bundles:
# - Task videos from RGB-D + instruction
# - 3D object flow extraction
# - Robot tracking targets

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY dream2flow-preparation-job/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY dream2flow-preparation-job/ .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Default command
CMD ["python", "entrypoint.py"]
