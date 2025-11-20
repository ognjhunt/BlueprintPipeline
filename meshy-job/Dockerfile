FROM python:3.10-slim

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --prefer-binary \
    requests

COPY run_meshy_from_assets.py /app/run_meshy_from_assets.py
COPY run_meshy.sh /app/run_meshy.sh

RUN chmod +x /app/run_meshy.sh

ENTRYPOINT ["/app/run_meshy.sh"]
