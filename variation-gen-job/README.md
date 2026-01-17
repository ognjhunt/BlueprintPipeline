# variation-gen-job

## Purpose / scope
Generates variation assets for the pipeline.

## Primary entrypoints
- `generate_variation_assets.py` job entrypoint.
- `run_variation_gen.sh` shell wrapper.
- `Dockerfile` container image definition.

## Required inputs / outputs
- **Inputs:** variation configuration, source assets, and manifests.
- **Outputs:** generated variation assets and metadata.

## Key environment variables
- Variables defining input/output locations and any service credentials.

## How to run locally
- Build the container: `docker build -t variation-gen-job .`
- Run the job: `python generate_variation_assets.py` or `./run_variation_gen.sh`.

