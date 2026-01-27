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
- `PIPELINE_ENV`: set to `production` to enforce placeholder disallow and production guardrails.
- `BP_ENV` and `PRODUCTION_MODE` are deprecated legacy production flags (removal after 2025-12-31).
- `DISALLOW_PLACEHOLDER_URDF`: set to `true` to fail when placeholder URDFs would be generated (enabled automatically in production).
- Variables defining input/output locations and pipeline configuration.

## How to run locally
- Build the container: `docker build -t interactive-job .`
- Run the job: `python run_interactive_assets.py`.
