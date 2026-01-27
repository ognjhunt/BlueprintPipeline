# meshy-job

## Purpose / scope
Generates or processes Meshy assets within the pipeline.

> **Note:** This job is intentionally a lightweight wrapper around Meshy tooling and
> API calls so the container stays focused on orchestration rather than bespoke logic.

## Primary entrypoints
- `run_meshy_from_assets.py` job entrypoint.
- `run_meshy.sh` shell wrapper.
- `Dockerfile` container image definition.

## Required inputs / outputs
- **Inputs:** asset layouts or references consumed by the Meshy job.
- **Outputs:** Meshy-generated assets and metadata.

## Key environment variables
- `MESHY_API_KEY`: Meshy API key for 3D generation.
- `MESHY_BASE_URL` or `MESHY_API_BASE`: Optional override for the Meshy API base URL (defaults to `https://api.meshy.ai`).

## How to run locally
- Build the container: `docker build -t meshy-job .`
- Run the job: `python run_meshy_from_assets.py` or `./run_meshy.sh`.
