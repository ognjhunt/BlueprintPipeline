# Episode Generation Job (Isaac Sim)

This job generates physics-validated, sensor-rich episodes using NVIDIA Isaac Sim
and Omniverse Replicator. It is designed for production data generation and
requires a GPU-backed Isaac Sim runtime.

## Requirements

- **Isaac Sim version:** 4.2.0 (matches `Dockerfile.isaacsim`).
- **GPU:** NVIDIA GPU required.
  - **Minimum:** 8 GB VRAM.
  - **Recommended:** T4, L4, or A100-class GPUs for stable performance.
- **Replicator extension:** `omni.replicator.core` must be available in the
  Isaac Sim environment.

## Production Environment Defaults

Production entrypoints (Docker/Kubernetes) explicitly enforce real sensor capture:

- `SENSOR_CAPTURE_MODE=isaac_sim`
- `USE_MOCK_CAPTURE=false`
- `DATA_QUALITY_LEVEL=production`
- `ISAAC_SIM_REQUIRED=true`
- `ALLOW_MOCK_DATA=false`

Mock capture is intended **only** for local development/testing.

## Lab Runs: Required Runtime + Fail-Fast Behavior

Labs should run episode generation inside an Isaac Sim runtime (4.2.0) with
Replicator enabled. Production flags (`DATA_QUALITY_LEVEL=production` or
`ISAAC_SIM_REQUIRED=true`) trigger fail-fast behavior: if Isaac Sim/Replicator
aren't available, the run raises immediately instead of falling back to mock
capture. `ALLOW_MOCK_DATA`/`ALLOW_MOCK_CAPTURE` are rejected in production mode,
so local tests must explicitly disable production flags and opt into
`SENSOR_CAPTURE_MODE=mock_dev` for mock data.

## Useful References

- `episode-generation-job/Dockerfile.isaacsim`
- `episode-generation-job/scripts/docker-entrypoint.sh`
- `docs/ISAAC_SIM_SETUP.md`
