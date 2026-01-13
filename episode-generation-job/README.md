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

Mock capture is intended **only** for local development/testing.

## Useful References

- `episode-generation-job/Dockerfile.isaacsim`
- `episode-generation-job/scripts/docker-entrypoint.sh`
- `docs/ISAAC_SIM_SETUP.md`
