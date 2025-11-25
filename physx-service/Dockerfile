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

# Our service implementation
COPY physx_service.py /app/physx_service.py

# Cloud Run expects the server to listen on this port
ENV PORT=8080

# Start the Flask app
CMD ["python", "physx_service.py"]
