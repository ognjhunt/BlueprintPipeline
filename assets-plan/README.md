# assets-plan

## Purpose / scope
Builds asset plans for scene assets used in the pipeline.

## Primary entrypoints
- `build_scene_assets.py` generates asset plans.
- `Dockerfile` container image definition.

## Required inputs / outputs
- **Inputs:** asset configuration, scene manifests, or source assets referenced by `build_scene_assets.py`.
- **Outputs:** generated asset plan artifacts.

## Key environment variables
- Environment variables used to locate input assets, output locations, or pipeline configuration.

## How to run locally
- Build the container: `docker build -t assets-plan .`
- Run the generator: `python build_scene_assets.py` (provide required inputs via env vars or CLI flags).

