FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY upsell-features-job/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy upsell features job
COPY upsell-features-job/ ./upsell-features-job/

# Copy shared tools
COPY tools/ ./tools/

# Set Python path
ENV PYTHONPATH=/app

# Run the job
CMD ["python", "upsell-features-job/generate_upsell_outputs.py"]
