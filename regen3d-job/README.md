# regen3d-job

## Purpose / scope
Adapter job for Regen3D processing within the pipeline.

## Primary entrypoints
- `regen3d_adapter_job.py` job entrypoint.

## Required inputs / outputs
- **Inputs:** Regen3D request payloads and asset references.
- **Outputs:** Regen3D-generated artifacts and metadata.

## Key environment variables
- Variables defining Regen3D endpoints, credentials, and input/output locations.

## How to run locally
- Run the adapter: `python regen3d_adapter_job.py` (set required env vars).

