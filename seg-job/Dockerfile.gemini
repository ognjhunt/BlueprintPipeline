FROM python:3.10-slim

# System deps for image processing
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps: Pillow, Google GenAI SDK
RUN pip install --no-cache-dir --prefer-binary \
    pillow \
    google-genai

# Job scripts
COPY run_gemini_inventory.py /app/run_gemini_inventory.py
COPY run_sam3.sh /app/run_sam3.sh

RUN chmod +x /app/run_sam3.sh

ENTRYPOINT ["/app/run_sam3.sh"]
