# isaac-lab-job

## Purpose / scope
Generates Isaac Lab tasks for the pipeline.

## Primary entrypoints
- `generate_isaac_lab_task.py` job entrypoint.
- `Dockerfile` container image definition.

## Required inputs / outputs
- **Inputs:** task configuration and asset references consumed by `generate_isaac_lab_task.py`.
- **Outputs:** generated Isaac Lab task definitions.

## Key environment variables
- Variables defining input/output locations and task configuration.

## How to run locally
- Build the container: `docker build -t isaac-lab-job .`
- Run the generator: `python generate_isaac_lab_task.py`.

