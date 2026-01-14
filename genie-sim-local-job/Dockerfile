# Genie Sim Local Execution Job Dockerfile
# Runs Genie Sim data collection locally when no API key is available.

FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY genie-sim-local-job/requirements.txt /app/requirements.txt
COPY genie-sim-local-job/requirements.lock /app/requirements.lock
RUN pip install --no-cache-dir -r requirements.txt -c requirements.lock

# Copy job code and shared tools
COPY genie-sim-submit-job/ /app/genie-sim-submit-job/
COPY genie-sim-export-job/ /app/genie-sim-export-job/
COPY tools/ /app/tools/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "/app/genie-sim-submit-job/submit_to_geniesim.py"]
