# usd-assembly-job

## Purpose / scope
Assembles USD scenes and converts assets for the pipeline.

## Primary entrypoints
- `assemble_scene.py` scene assembly entrypoint.
- `build_scene_usd.py` USD build helper.
- `glb_to_usd.py` conversion helper.
- `run_usd.sh` shell wrapper.
- `Dockerfile` container image definition.

## Required inputs / outputs
- **Inputs:** scene layouts, asset references, and GLB/USD source data.
- **Outputs:** assembled USD scenes and supporting artifacts.

## Key environment variables
- Variables defining input/output locations and conversion settings.

## How to run locally
- Build the container: `docker build -t usd-assembly-job .`
- Run the job: `python assemble_scene.py` or `./run_usd.sh`.

