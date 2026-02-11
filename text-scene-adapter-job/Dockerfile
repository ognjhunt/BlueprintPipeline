FROM python:3.11-slim

WORKDIR /app

COPY text-scene-adapter-job/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY tools/ /app/tools/
COPY text-scene-adapter-job/adapt_text_scene.py /app/text-scene-adapter-job/adapt_text_scene.py

ENTRYPOINT ["python", "/app/text-scene-adapter-job/adapt_text_scene.py"]
