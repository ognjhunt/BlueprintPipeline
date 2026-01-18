FROM python:3.11-slim

WORKDIR /app

COPY infrastructure/secret-rotation/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY tools/secrets/secret_manager.py /app/secret_manager.py
COPY infrastructure/secret-rotation/rotate_secrets.py /app/rotate_secrets.py

ENV PYTHONUNBUFFERED=1

CMD ["python", "rotate_secrets.py"]
