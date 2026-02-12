FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY asset-replication-job/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY asset-replication-job/process_replication_queue.py .

CMD ["python", "process_replication_queue.py"]
