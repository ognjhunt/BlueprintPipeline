FROM python:3.10-slim

# System deps for NumPy / OpenCV / Pillow
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps: NumPy, OpenCV, Pillow, Google GenAI SDK
RUN pip install --no-cache-dir --prefer-binary \
    numpy \
    opencv-python-headless \
    pillow \
    google-genai

# Job scripts (keep old script for backward compatibility)
COPY run_multiview_from_layout.py /app/run_multiview_from_layout.py
COPY run_multiview_gemini_generative.py /app/run_multiview_gemini_generative.py
COPY run_multiview.sh /app/run_multiview.sh

RUN chmod +x /app/run_multiview.sh

ENTRYPOINT ["/app/run_multiview.sh"]
