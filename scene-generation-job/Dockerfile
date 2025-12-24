# Scene Generation Job Container
#
# Generates scene images using Gemini 3.0 Pro Image for 3D reconstruction pipeline.

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY generate_scene_images.py .
COPY archetypes/ archetypes/

# Create output directory
RUN mkdir -p /tmp/scene_generation

# Default environment variables
ENV SCENES_PER_RUN=10
ENV DRY_RUN=false

# Run the job
CMD ["python", "generate_scene_images.py"]
