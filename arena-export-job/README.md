# arena-export-job

## Purpose / scope
Builds and runs the Arena export job, producing analytics and artifacts for Arena export pipelines.

## Primary entrypoints
- `arena_export_job.py` main job entrypoint.
- `Dockerfile` container image definition.
- `default_*.py` default analyzers/handlers used by the job.

## Required inputs / outputs
- **Inputs:** job configuration, scene/simulation artifacts, and any analytics configuration consumed by `arena_export_job.py`.
- **Outputs:** Arena export artifacts and analytics reports emitted by the job.

## Key environment variables
- Job configuration variables (e.g., pipeline/asset identifiers, storage locations) consumed by `arena_export_job.py`.

## How to run locally
- Build the container: `docker build -t arena-export-job .`
- Run the job: `python arena_export_job.py` (set required env vars and provide any needed input files).

