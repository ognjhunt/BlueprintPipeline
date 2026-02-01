FROM python:3.11-slim

# System deps for NumPy
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
RUN pip install --no-cache-dir --prefer-binary \
    numpy

# Job scripts
COPY run_scale_from_layout.py /app/run_scale_from_layout.py
COPY run_scale.sh /app/run_scale.sh

RUN chmod +x /app/run_scale.sh

ENTRYPOINT ["/app/run_scale.sh"]
