FROM python:3.11-slim

WORKDIR /app

COPY text-scene-gen-job/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY tools/ /app/tools/
COPY text-scene-gen-job/generate_text_scene.py /app/text-scene-gen-job/generate_text_scene.py

ENTRYPOINT ["python", "/app/text-scene-gen-job/generate_text_scene.py"]
