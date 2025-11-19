FROM python:3.10-slim

# System deps for NumPy / OpenCV
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
RUN pip install --no-cache-dir --prefer-binary \
    numpy \
    opencv-python-headless

# Job scripts
COPY run_layout_from_da3.py /app/run_layout_from_da3.py
COPY run_layout.sh /app/run_layout.sh

RUN chmod +x /app/run_layout.sh

ENTRYPOINT ["/app/run_layout.sh"]
