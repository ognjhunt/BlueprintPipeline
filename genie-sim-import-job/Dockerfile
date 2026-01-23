FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY genie-sim-import-job/requirements.lock ./requirements.lock
RUN pip install --no-cache-dir -r requirements.lock && pip check

# Copy shared tools and dependencies
COPY tools/ ./tools/
COPY genie-sim-export-job/ ./genie-sim-export-job/
COPY episode-generation-job/ ./episode-generation-job/

# Copy job files
COPY genie-sim-import-job/ ./genie-sim-import-job/

# Set working directory to job
WORKDIR /app/genie-sim-import-job

# Entry point
CMD ["python", "import_from_geniesim.py"]
