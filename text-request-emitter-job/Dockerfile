FROM python:3.11-slim

WORKDIR /app

COPY text-request-emitter-job/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY tools/ ./tools/
COPY configs/ ./configs/
COPY text-request-emitter-job/ ./text-request-emitter-job/

ENV PYTHONUNBUFFERED=1

CMD ["python", "text-request-emitter-job/emit_text_requests.py"]
