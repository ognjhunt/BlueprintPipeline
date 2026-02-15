# syntax=docker/dockerfile:1
# infinigen-service/Dockerfile
#
# Infinigen articulated asset generation backend.
# Repo: https://github.com/princeton-vl/infinigen
#
# Exposes a small HTTP API returning a zip payload (URDF + meshes) for interactive-job.
#
# NOTE: Infinigen has additional dependencies; this Dockerfile is a best-effort
#       packaging that avoids runtime downloads by baking the repo into the image.

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

ENV INFINIGEN_ROOT=/opt/infinigen
RUN git clone https://github.com/princeton-vl/infinigen.git "${INFINIGEN_ROOT}"

# Install Infinigen into the image (editable install).
RUN pip install --no-cache-dir -e "${INFINIGEN_ROOT}"

RUN pip install --no-cache-dir flask gunicorn

WORKDIR /app
COPY infinigen_service.py /app/infinigen_service.py

ENV PORT=8080
ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f\"http://localhost:{os.environ.get('PORT','8080')}/\", timeout=5).read()" || exit 1

CMD ["gunicorn", \
     "--bind", "0.0.0.0:8080", \
     "--workers", "1", \
     "--threads", "2", \
     "--timeout", "1200", \
     "--graceful-timeout", "60", \
     "--keep-alive", "30", \
     "--log-level", "info", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--capture-output", \
     "infinigen_service:app"]

