FROM python:3.11-slim

# System deps for NumPy / OpenCV
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
RUN pip install --no-cache-dir --prefer-binary \
    numpy \
    opencv-python-headless \
    pyyaml

# Job scripts
COPY run_objects_from_layout.py /app/run_objects_from_layout.py
COPY run_objects.sh /app/run_objects.sh

RUN chmod +x /app/run_objects.sh

ENTRYPOINT ["/app/run_objects.sh"]
