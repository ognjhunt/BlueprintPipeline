FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY scene-batch-job/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy scene batch job
COPY scene-batch-job/ ./scene-batch-job/

# Copy shared tools
COPY tools/ ./tools/

# Set Python path
ENV PYTHONPATH=/app

# Run the job
CMD ["python", "scene-batch-job/entrypoint.py"]
