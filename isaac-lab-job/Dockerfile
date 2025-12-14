# Isaac Lab Task Generation Job
# Generates RL training packages from BlueprintPipeline scenes

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    numpy \
    pyyaml

# Copy tools and job
COPY tools/ /app/tools/
COPY policy_configs/ /app/policy_configs/
COPY isaac-lab-job/ /app/isaac-lab-job/

# Set Python path
ENV PYTHONPATH=/app

# Entry point
CMD ["python", "/app/isaac-lab-job/generate_isaac_lab_task.py"]
