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

## How to run locally
- Build the container: `docker build -t genie-sim-submit-job .`
- Run the submitter: `python submit_to_geniesim.py`.

