# Smart Placement Engine
# Intelligent asset-to-placement-region matching with collision awareness

FROM python:3.11-slim

LABEL maintainer="Blueprint Team"
LABEL description="Smart Placement Engine with AI-powered region detection and collision awareness"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Install gcsfuse for GCS bucket mounting
RUN echo "deb https://packages.cloud.google.com/apt gcsfuse-$(lsb_release -cs) main" \
    | tee /etc/apt/sources.list.d/gcsfuse.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
    && apt-get update && apt-get install -y gcsfuse \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create mount point for GCS
RUN mkdir -p /mnt/gcs

# Create entrypoint script
RUN cat > /app/entrypoint.sh << 'EOF'
#!/bin/bash
set -e

# Mount GCS bucket if BUCKET is set
if [ -n "$BUCKET" ]; then
    echo "Mounting GCS bucket: $BUCKET"
    gcsfuse --implicit-dirs "$BUCKET" /mnt/gcs
    trap "fusermount -u /mnt/gcs || true" EXIT
fi

# Run the smart placement engine
python run_smart_placement.py "$@"
EOF

RUN chmod +x /app/entrypoint.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import run_smart_placement; print('OK')" || exit 1

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
