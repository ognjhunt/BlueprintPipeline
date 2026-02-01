# Genie Sim GPU Runtime (Isaac Sim)

This guide documents the runtime prerequisites and environment variables required
for running the GPU-capable Genie Sim job image built from `genie-sim-gpu-job/`.

## Prerequisites

1. **NVIDIA GPU + Drivers**
   - Ensure the host has a supported NVIDIA GPU (T4/A10/A100/etc.).
   - Install the NVIDIA Container Toolkit so containers can access GPUs.

2. **Isaac Sim Container Access**
   - The image is built on `nvcr.io/nvidia/isaac-sim:5.1.0`.
   - Configure NGC access if you pull from NVIDIA NGC.

3. **GKE (required for `genie-sim-gpu-job`)**
   - The Genie Sim GPU job is dispatched to **GKE** (not Cloud Run) for GPU access.
   - A GPU node pool is required (example: `nvidia-tesla-t4`).
   - Install the NVIDIA device plugin on the cluster.
   - Provide a service account secret for GCS access.
   - See `k8s/genie-sim-gpu-job.yaml` and `scripts/deploy-genie-sim-gpu-job.sh`.

## Required Environment Variables

These variables map to the same defaults referenced in
`tools/geniesim_adapter/deployment/README.md` and the local framework:

- `ISAAC_SIM_PATH`: Path to Isaac Sim (default: `/isaac-sim`).
- `GENIESIM_ROOT`: Path to the Genie Sim repository checkout (default: `/opt/geniesim`).
- `GENIESIM_HOST`: gRPC host for the Genie Sim server (default: `localhost`).
- `GENIESIM_PORT`: gRPC port for the Genie Sim server (defaults to the adapter port; set for non-local deployments).
- `GENIESIM_VALIDATE_FRAMES`: Validate recorded frames before saving (set to `1` in production).
- `GENIESIM_FAIL_ON_FRAME_VALIDATION`: Fail episodes when frame validation errors exist (set to `1` in production).

## Optional Runtime Inputs

- `BUCKET`: GCS bucket for inputs/outputs.
- `SCENE_ID`: Scene identifier to process.
- `OUTPUT_PREFIX`: Where to write episodes (`scenes/<scene>/episodes`).
- `ROBOT_TYPE`: Robot type (`franka`, `g2`, `ur10`, etc.).

## Example (GKE)

```bash
export SCENE_ID="kitchen_001"
export BUCKET="my-scenes-bucket"
./scripts/deploy-genie-sim-gpu-job.sh
```
