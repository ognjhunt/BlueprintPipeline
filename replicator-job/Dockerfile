# Replicator Job - Lightweight container for generating Replicator bundles
# Uses Gemini API for scene analysis - no heavy ML models needed

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install gcsfuse for GCS mounting
RUN curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
    && echo "deb https://packages.cloud.google.com/apt gcsfuse-buster main" > /etc/apt/sources.list.d/gcsfuse.list \
    && apt-get update \
    && apt-get install -y gcsfuse \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

# Copy application code
COPY generate_replicator_bundle.py .
COPY utils/ ./utils/ 2>/dev/null || true
COPY policy_configs/ ./policy_configs/ 2>/dev/null || true
COPY templates/ ./templates/ 2>/dev/null || true

# Create mount point for GCS
RUN mkdir -p /mnt/gcs

# Entry point script
COPY <<'EOF' /app/entrypoint.sh
#!/bin/bash
set -e

echo "[REPLICATOR] Starting replicator-job..."
echo "[REPLICATOR] Scene ID: ${SCENE_ID}"
echo "[REPLICATOR] Bucket: ${BUCKET}"

# Mount GCS bucket using gcsfuse
if [ -n "${BUCKET}" ]; then
    echo "[REPLICATOR] Mounting GCS bucket: ${BUCKET}"
    gcsfuse --implicit-dirs "${BUCKET}" /mnt/gcs
    echo "[REPLICATOR] GCS mounted successfully"
fi

# Run the main script
python generate_replicator_bundle.py

# Unmount GCS
if mountpoint -q /mnt/gcs; then
    echo "[REPLICATOR] Unmounting GCS..."
    fusermount -u /mnt/gcs || true
fi

echo "[REPLICATOR] Job completed"
EOF

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
