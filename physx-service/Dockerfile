# physx-service/Dockerfile
FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install FastAPI + Uvicorn
RUN pip install --no-cache-dir fastapi uvicorn[standard]

COPY main.py /app/main.py

# Cloud Run listens on $PORT; default to 8080.
ENV PORT=8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
