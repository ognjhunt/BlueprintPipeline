# Arena Export Job Dockerfile
# Converts Blueprint scenes to Isaac Lab-Arena format

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY arena-export-job/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy job code and shared tools
COPY arena-export-job/ /app/arena-export-job/
COPY tools/ /app/tools/
COPY policy_configs/ /app/policy_configs/

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run in environment variable mode by default
ENTRYPOINT ["python", "/app/arena-export-job/arena_export_job.py", "--env-mode"]
