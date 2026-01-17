# scale-job

## Purpose / scope
Runs Scale-related asset processing jobs for the pipeline.

## Primary entrypoints
- `run_scale_from_layout.py` job entrypoint.
- `run_scale.sh` shell wrapper.
- `Dockerfile` container image definition.

## Required inputs / outputs
- **Inputs:** asset layouts or configuration consumed by the Scale job.
- **Outputs:** processed Scale assets and metadata.

## Key environment variables
- Variables defining input/output locations and any Scale service credentials.

## How to run locally
- Build the container: `docker build -t scale-job .`
- Run the job: `python run_scale_from_layout.py` or `./run_scale.sh`.

