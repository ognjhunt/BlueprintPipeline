FROM python:3.11-slim

WORKDIR /app

COPY genie-sim-import-webhook/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY genie-sim-import-webhook/main.py /app/main.py

ENV PYTHONUNBUFFERED=1

CMD ["gunicorn", "-b", ":8080", "main:app"]
