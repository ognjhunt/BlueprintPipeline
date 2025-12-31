FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY episode-generation-job/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy episode generation job
COPY episode-generation-job/ ./episode-generation-job/

# Copy shared tools
COPY tools/ ./tools/

# Copy DWM modules for scene analysis (optional)
COPY dwm-preparation-job/ ./dwm_preparation_job/

# Set Python path
ENV PYTHONPATH=/app

# Run the job
CMD ["python", "-m", "episode-generation-job.generate_episodes"]
