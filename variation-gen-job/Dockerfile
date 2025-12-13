# syntax=docker/dockerfile:1
# Variation Asset Generator - Generates reference images for domain randomization assets
#
# This lightweight container uses Gemini 3.0 Pro Image (Nano Banana Pro) to generate reference images
# for variation assets identified by replicator-job. The generated images can be
# processed by downstream jobs to create 3D models.

FROM python:3.10-slim

# System deps for Pillow image handling
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Job scripts
COPY generate_variation_assets.py /app/generate_variation_assets.py
COPY run_variation_gen.sh /app/run_variation_gen.sh

RUN chmod +x /app/run_variation_gen.sh

ENTRYPOINT ["/app/run_variation_gen.sh"]
