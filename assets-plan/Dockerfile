FROM python:3.11-slim

# Create non-root user
RUN useradd -m appuser
WORKDIR /app

# Copy code
COPY build_scene_assets.py .

# No external deps; stdlib only
USER appuser

# Run the planner
ENTRYPOINT ["python", "-u", "build_scene_assets.py"]
