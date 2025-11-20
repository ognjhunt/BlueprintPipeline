FROM python:3.10-slim

# Install system dependencies for basic file operations if needed
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY run_interactive_assets.py /app/run_interactive_assets.py
COPY run_interactive.sh /app/run_interactive.sh

RUN chmod +x /app/run_interactive.sh

ENTRYPOINT ["/app/run_interactive.sh"]