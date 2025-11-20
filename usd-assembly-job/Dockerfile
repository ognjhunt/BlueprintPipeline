FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir numpy

COPY build_scene_usd.py /app/build_scene_usd.py
COPY run_usd.sh /app/run_usd.sh

RUN chmod +x /app/run_usd.sh

ENTRYPOINT ["/app/run_usd.sh"]
