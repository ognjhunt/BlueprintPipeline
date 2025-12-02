FROM python:3.10-slim

# Basic build tools for trimesh (via numpy); keep it minimal.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install just what we need to run a tiny HTTP service and build a cube GLB.
RUN pip install --no-cache-dir \
    flask \
    trimesh \
    numpy

# Copy both the orchestrator and the service (for reference)
COPY run_interactive_assets.py /app/run_interactive_assets.py
COPY physx_service.py /app/physx_service.py

# Run the actual job orchestrator
CMD ["python", "run_interactive_assets.py"]
