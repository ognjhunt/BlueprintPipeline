# Genie Sim Server Deployment (Local)

This directory provides a **local deployment path** for the Genie Sim server that is
compatible with the BlueprintPipeline gRPC adapter in `tools/geniesim_adapter/`.

## Quick Start

```bash
# From the repo root
cd tools/geniesim_adapter/deployment

# Bootstrap Genie Sim runtime + health check
./bootstrap_geniesim_runtime.sh

# Install/clone Genie Sim
./install_geniesim.sh

# Export required environment variables
export GENIESIM_ROOT=${GENIESIM_ROOT:-/opt/geniesim}
export ISAAC_SIM_PATH=${ISAAC_SIM_PATH:-/isaac-sim}
export GENIESIM_HOST=${GENIESIM_HOST:-localhost}
export GENIESIM_PORT=${GENIESIM_PORT:-50051}

# Launch the server inside Isaac Sim (example)
${ISAAC_SIM_PATH}/python.sh \
  ${GENIESIM_ROOT}/source/data_collection/scripts/data_collector_server.py \
  --headless \
  --port ${GENIESIM_PORT}
```

## Environment Variables

The local framework relies on these variables:

- `GENIESIM_ROOT`: Path to the Genie Sim repo checkout (default: `/opt/geniesim`)
- `ISAAC_SIM_PATH`: Path to the Isaac Sim installation (default: `/isaac-sim`)
- `GENIESIM_HOST`: gRPC host for the Genie Sim server (default: `localhost`)
- `GENIESIM_PORT`: gRPC port for the Genie Sim server (default: `50051`)
- `GENIESIM_CLEANUP_TMP`: Remove Genie Sim temp directories after a run completes
  (default: `1` for local/development, `0` in production)

## Notes

- The install script is designed for **local setups** and can be customized to match
  your environment (CUDA/driver requirements vary across hosts).
- If you already have Genie Sim cloned, set `GENIESIM_ROOT` and re-run the script to
  install dependencies.
- The bootstrap script starts the server (unless `GENIESIM_START_SERVER=0`) and runs
  `python -m tools.geniesim_adapter.geniesim_healthcheck` for a readiness check.
