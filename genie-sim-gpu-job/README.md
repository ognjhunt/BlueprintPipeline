# Genie Sim GPU Job Image

This image packages **NVIDIA Isaac Sim** with the **Genie Sim** repository installed for
GPU-backed local execution. It is designed to run the local Genie Sim submission flow
using the expected paths from `tools/geniesim_adapter/deployment/README.md`:

- `ISAAC_SIM_PATH=/isaac-sim`
- `GENIESIM_ROOT=/opt/geniesim`

> **Note:** This job is intentionally a lightweight wrapper around existing Genie Sim
> tooling, keeping the container focused on packaging Isaac Sim and invoking the
> upstream entrypoints without extra orchestration logic.

## Build

Build arguments:

- `ISAAC_SIM_VERSION` (default `5.1.0`) sets the Isaac Sim base image tag.
- `GENIESIM_REPO` (default `https://github.com/AgibotTech/genie_sim.git`) sets the Genie Sim repo.
- `GENIESIM_REF` (default pinned in `genie-sim-gpu-job/GENIESIM_REF`) sets the Genie Sim ref.

```bash
# From repo root
export GENIESIM_REPO=https://github.com/AgibotTech/genie_sim.git
export GENIESIM_REF=main
export ISAAC_SIM_VERSION=5.1.0

docker build \
  -f genie-sim-gpu-job/Dockerfile \
  --build-arg ISAAC_SIM_VERSION=${ISAAC_SIM_VERSION} \
  --build-arg GENIESIM_REPO=${GENIESIM_REPO} \
  --build-arg GENIESIM_REF=${GENIESIM_REF} \
  -t gcr.io/$PROJECT_ID/blueprint-genie-sim:isaacsim .
```

Example override for a newer Isaac Sim base image:

```bash
docker build \
  -f genie-sim-gpu-job/Dockerfile \
  --build-arg ISAAC_SIM_VERSION=5.1.0 \
  -t gcr.io/$PROJECT_ID/blueprint-genie-sim:isaacsim .
```

## Runtime Environment Variables

The job expects the same runtime variables as `genie-sim-local-job` plus Genie Sim
connection details:

- `ISAAC_SIM_PATH` (default `/isaac-sim`)
- `GENIESIM_ROOT` (default `/opt/geniesim`)
- `GENIESIM_HOST` (default `localhost`)
- `GENIESIM_PORT` (default `50051`)
- `GENIESIM_GPU_PREALLOCATE_MB` (optional, integer MB). When set to a value > 0, the
  entrypoint runs a GPU warmup step to preallocate the requested memory.
- `GENIESIM_GPU_WARMUP_STRICT` (optional, `1`/`true`). When enabled, the warmup step
  exits non-zero on failure; otherwise failures are logged and ignored.

## Notes

- GPU runtime and NVIDIA container toolkit are required.
- The container runs `submit_to_geniesim.py` via `python.sh` from Isaac Sim.
- The warmup step uses `torch` if available; if not present or CUDA is unavailable, it
  logs a warning and continues.
