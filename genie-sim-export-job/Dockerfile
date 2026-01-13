FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY genie-sim-export-job/requirements.txt ./requirements.txt
COPY genie-sim-export-job/requirements.lock ./requirements.lock
RUN pip install --no-cache-dir -r requirements.txt -c requirements.lock

# Copy tools directory (geniesim_adapter module)
COPY tools/ ./tools/

# Copy job files
COPY genie-sim-export-job/ ./genie-sim-export-job/

# Set working directory to job
WORKDIR /app/genie-sim-export-job

# Entry point
CMD ["python", "export_to_geniesim.py"]
