# genie-sim-submit-job

## Purpose / scope
Submits jobs to GenieSim and coordinates submission metadata.

## Primary entrypoints
- `submit_to_geniesim.py` job entrypoint.
- `Dockerfile` container image definition.

## Required inputs / outputs
- **Inputs:** submission payloads, job configuration, and credentials used by GenieSim.
- **Outputs:** submission status and any emitted tracking metadata.

## Key environment variables
- Credentials and endpoint configuration for GenieSim, plus any pipeline configuration variables.
- Export controls:
  - `LEROBOT_EXPORT_FORMAT=lerobot_v3` (recommended for VLA/VLM consumers)
  - `ENABLE_RLDS_EXPORT=1` (also export RLDS TFRecords to `output_dir/rlds/`)
- P1 domain randomization (run-time; requires the server patch applied by `bootstrap_geniesim_runtime.sh`):
  - `GENIESIM_DR_LEVEL=essential`
  - `GENIESIM_DR_SEED=0`
  - Optional: `GENIESIM_DR_ENABLE_OBS_NOISE=1`, `GENIESIM_DR_RGB_NOISE_STD=3.0`, `GENIESIM_DR_DEPTH_NOISE_STD_M=0.001`
- P1 depth/point-cloud sidecars (export-time):
  - `LEROBOT_EXPORT_INCLUDE_DEPTH=1`
  - `LEROBOT_EXPORT_INCLUDE_POINT_CLOUD=1`
  - `LEROBOT_POINT_CLOUD_MAX_POINTS=2048`
  - `LEROBOT_POINT_CLOUD_FRAME=world`

## How to run locally
- Build the container: `docker build -t genie-sim-submit-job .`
- Run the submitter: `python submit_to_geniesim.py`.
