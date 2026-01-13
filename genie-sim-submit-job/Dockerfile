# Genie Sim Submit Job Dockerfile
# Submits Genie Sim generation jobs from export bundles

FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY genie-sim-submit-job/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy job code and shared tools
COPY genie-sim-submit-job/ /app/genie-sim-submit-job/
COPY genie-sim-export-job/ /app/genie-sim-export-job/
COPY tools/ /app/tools/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "/app/genie-sim-submit-job/submit_to_geniesim.py"]
