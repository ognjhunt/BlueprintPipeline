FROM python:3.10-slim

WORKDIR /app

# Basic deps: numpy for simple math, google-genai for Gemini
RUN pip install --no-cache-dir \
    numpy \
    google-genai

# Our simready prep script + entrypoint
COPY prepare_simready_assets.py /app/prepare_simready_assets.py
COPY run_simready.sh /app/run_simready.sh

RUN chmod +x /app/run_simready.sh

ENTRYPOINT ["/app/run_simready.sh"]
