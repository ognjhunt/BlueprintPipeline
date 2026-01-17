# objects-job

## Purpose / scope
Generates object assets or layouts for the pipeline.

## Primary entrypoints
- `run_objects_from_layout.py` job entrypoint.
- `run_objects.sh` shell wrapper.
- `Dockerfile` container image definition.

## Required inputs / outputs
- **Inputs:** object layout definitions and asset configuration.
- **Outputs:** generated object assets and related metadata.

## Key environment variables
- Variables defining input/output locations and pipeline configuration.

## How to run locally
- Build the container: `docker build -t objects-job .`
- Run the job: `python run_objects_from_layout.py` or `./run_objects.sh`.

