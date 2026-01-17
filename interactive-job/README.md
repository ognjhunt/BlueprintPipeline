# interactive-job

## Purpose / scope
Runs interactive asset processing for the pipeline.

## Primary entrypoints
- `run_interactive_assets.py` job entrypoint.
- `heuristic_articulation.py` heuristics support module.
- `Dockerfile` container image definition.

## Required inputs / outputs
- **Inputs:** asset layouts or scene descriptions used by `run_interactive_assets.py`.
- **Outputs:** processed interactive assets and associated metadata.

## Key environment variables
- Variables defining input/output locations and pipeline configuration.

## How to run locally
- Build the container: `docker build -t interactive-job .`
- Run the job: `python run_interactive_assets.py`.

