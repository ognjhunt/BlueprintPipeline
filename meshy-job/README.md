# meshy-job

## Purpose / scope
Generates or processes Meshy assets within the pipeline.

## Primary entrypoints
- `run_meshy_from_assets.py` job entrypoint.
- `run_meshy.sh` shell wrapper.
- `Dockerfile` container image definition.

## Required inputs / outputs
- **Inputs:** asset layouts or references consumed by the Meshy job.
- **Outputs:** Meshy-generated assets and metadata.

## Key environment variables
- Variables defining input/output locations and any Meshy service configuration.

## How to run locally
- Build the container: `docker build -t meshy-job .`
- Run the job: `python run_meshy_from_assets.py` or `./run_meshy.sh`.

