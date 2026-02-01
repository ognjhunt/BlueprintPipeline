# Episode Generation Job (Isaac Sim)

This job generates physics-validated, sensor-rich episodes using NVIDIA Isaac Sim
and Omniverse Replicator. It is designed for production data generation and
requires a GPU-backed Isaac Sim runtime.

## Requirements

- **Isaac Sim version:** 5.1.0 (matches `Dockerfile.isaacsim`).
- **GPU:** NVIDIA GPU required.
  - **Minimum:** 8 GB VRAM.
  - **Recommended:** T4, L4, or A100-class GPUs for stable performance.
- **Replicator extension:** `omni.replicator.core` must be available in the
  Isaac Sim environment.

## Production Environment Defaults

Production entrypoints (Docker/Kubernetes) explicitly enforce real sensor capture:

- `PIPELINE_ENV=production`
- `SENSOR_CAPTURE_MODE=isaac_sim`
- `USE_MOCK_CAPTURE=false`
- `DATA_QUALITY_LEVEL=production`
- `ISAAC_SIM_REQUIRED=true`
- `ALLOW_MOCK_DATA=false`

Mock capture is intended **only** for local development/testing.

## Camera Calibration Requirement

LeRobot exports that include RGB observations will validate camera calibration
matrices by default (inferred from `include_images` and the data pack tier).
For development-only exports where calibration data is intentionally absent,
disable the requirement with:

```bash
export REQUIRE_CAMERA_CALIBRATION=false
```

## Lab Runs: Required Runtime + Fail-Fast Behavior

Labs should run episode generation inside an Isaac Sim runtime (5.1.0) with
Replicator enabled. Production toggles (`PIPELINE_ENV=production` plus
`DATA_QUALITY_LEVEL=production` or `ISAAC_SIM_REQUIRED=true`) trigger fail-fast
behavior: if Isaac Sim/Replicator aren't available, the run raises immediately
instead of falling back to mock capture. `ALLOW_MOCK_DATA`/`ALLOW_MOCK_CAPTURE`
are rejected in production mode, so local tests must explicitly disable
production mode and opt into
`SENSOR_CAPTURE_MODE=mock_dev` for mock data.

For labs staging/production-quality runs, cuRobo collision planning is required:

- **cuRobo** (NVIDIA cuRobo Python package)
- **CUDA** runtime + NVIDIA drivers
- **PyTorch with CUDA support**

Missing these dependencies causes a hard failure in labs/production runs to
avoid generating collision-unsafe trajectories.

## Optional Firebase Uploads

The job can optionally upload generated episodes to Firebase Storage after the
dataset export step.

Set the following environment variables to enable uploads:

- `ENABLE_FIREBASE_UPLOAD=true`
- `FIREBASE_STORAGE_BUCKET` (e.g., `blueprint-8c1ca.appspot.com`)
- `FIREBASE_SERVICE_ACCOUNT_JSON` **or** `FIREBASE_SERVICE_ACCOUNT_PATH`
- `FIREBASE_UPLOAD_PREFIX` (optional, default: `datasets`)

## Useful References

- `episode-generation-job/Dockerfile.isaacsim`
- `episode-generation-job/scripts/docker-entrypoint.sh`
- `docs/ISAAC_SIM_SETUP.md`
