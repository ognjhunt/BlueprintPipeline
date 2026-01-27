FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY regen3d-job/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy regen3d job
COPY regen3d-job/ ./regen3d-job/

# Copy shared tools
COPY tools/ ./tools/

# Set Python path
ENV PYTHONPATH=/app

# Run the job
CMD ["python", "regen3d-job/regen3d_adapter_job.py"]
