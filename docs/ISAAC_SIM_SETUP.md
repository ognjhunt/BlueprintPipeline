# Isaac Sim Environment Setup

This guide covers setting up NVIDIA Isaac Sim for use with BlueprintPipeline's episode generation capabilities.

## Prerequisites

- NVIDIA GPU with RTX support (RTX 2060 or higher recommended)
- Docker with NVIDIA Container Toolkit
- 32GB+ RAM recommended (16GB minimum)
- Ubuntu 20.04/22.04 or Windows 10/11 with WSL2

## Episode Generation Runtime Requirements (Production)

Episode generation **must** run inside an Isaac Sim runtime with Replicator enabled.
We standardize on the NVIDIA NGC container image below for production runs:

- **Container image**: `nvcr.io/nvidia/isaac-sim:2024.1.0` (Isaac Sim 2024.1.0+)
- **Required extensions**: `omni.isaac.core`, `omni.physx`, `omni.replicator.core`
- **GPU**: NVIDIA RTX/Tesla-class GPU (T4/L4/A10/A100) with **16GB+ VRAM**
- **Driver/tooling**: NVIDIA driver + NVIDIA Container Toolkit installed

If Replicator is disabled or Isaac Sim is unavailable, production runs will fail
fast to prevent generating mock (non-trainable) data.

## Option 1: NVIDIA NGC Container (Recommended)

The fastest way to get started is using NVIDIA's pre-built Isaac Sim container from NGC.

### Docker Compose (BlueprintPipeline)

When using `docker-compose.isaacsim.yaml`, you can run Compose from outside the repo root by
setting `PROJECT_ROOT` to the BlueprintPipeline directory:

```bash
PROJECT_ROOT=/path/to/BlueprintPipeline docker compose -f docker-compose.isaacsim.yaml up episode-generation
```

### Pull and Run Isaac Sim Container

```bash
# Pull Isaac Sim container (2024.1.0 or later)
docker pull nvcr.io/nvidia/isaac-sim:2024.1.0

# Run with BlueprintPipeline mounted
docker run --gpus all -it \
  -v $(pwd):/workspace/BlueprintPipeline \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --network=host \
  nvcr.io/nvidia/isaac-sim:2024.1.0 \
  bash

# Inside container
cd /workspace/BlueprintPipeline
/isaac-sim/python.sh tools/run_local_pipeline.py --help
```

### Enable GUI (Optional)

If you need the Isaac Sim GUI for debugging:

```bash
# On host, allow X11 connections
xhost +local:docker

# Run with GUI support
docker run --gpus all -it \
  -v $(pwd):/workspace/BlueprintPipeline \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --network=host \
  nvcr.io/nvidia/isaac-sim:2024.1.0 \
  bash

# Inside container, launch Isaac Sim
/isaac-sim/isaac-sim.sh
```

## Option 2: Custom Dockerfile

For production deployments, build a custom image with BlueprintPipeline baked in:

```dockerfile
FROM nvcr.io/nvidia/isaac-sim:2024.1.0

# Set working directory
WORKDIR /workspace

# Copy BlueprintPipeline
COPY . /workspace/BlueprintPipeline

# Install pinned Python dependencies
RUN /isaac-sim/python.sh -m pip install --upgrade pip && \
    /isaac-sim/python.sh -m pip install -r /workspace/BlueprintPipeline/tools/requirements-pins.txt

# Install critical gap fixes (pinned)
RUN /isaac-sim/python.sh -m pip install -r /workspace/BlueprintPipeline/requirements-critical-gaps.txt

# Set entrypoint to Isaac Sim Python
ENTRYPOINT ["/isaac-sim/python.sh"]

# Default command runs local pipeline
CMD ["/workspace/BlueprintPipeline/tools/run_local_pipeline.py"]
```

### Build and Run

```bash
# Build custom image
docker build -t blueprint-isaac-sim:latest .

# Run episode generation
docker run --gpus all \
  -e PRODUCTION_MODE=true \
  -e SCENE_ID=kitchen_001 \
  -e NUM_EPISODES=100 \
  -v $(pwd)/output:/output \
  blueprint-isaac-sim:latest \
  /workspace/BlueprintPipeline/episode-generation-job/generate_episodes.py
```

## Verifying Isaac Sim Integration

After starting your container, verify that Isaac Sim and Replicator are available:

```bash
# Test Isaac Sim availability
/isaac-sim/python.sh -c "from episode_generation_job.isaac_sim_integration import is_isaac_sim_available; print(is_isaac_sim_available())"

# Expected output: True

# Test Replicator availability
/isaac-sim/python.sh -c "from episode_generation_job.isaac_sim_integration import is_replicator_available; print(is_replicator_available())"

# Expected output: True

# Test full integration
/isaac-sim/python.sh -c "
from episode_generation_job.isaac_sim_integration import IsaacSimIntegration
integration = IsaacSimIntegration()
print(f'Isaac Sim available: {integration.is_available()}')
print(f'Replicator available: {integration.has_replicator()}')
"
```

## Staging E2E Validation (Labs pre-production)

Labs should run the staging harness against **real** 3D-RE-GEN reconstructions
inside Isaac Sim before a production rollout. This exercises the real pipeline
handoff and confirms USD scenes load in Isaac Sim with the live backend.

Staging checklist (labs pre-production):
- Confirm Particulate availability via `PARTICULATE_ENDPOINT`, or run a local Particulate
  instance (`PARTICULATE_MODE=local`) with an approved `PARTICULATE_LOCAL_MODEL`.
Staging runs also require collision-aware planning dependencies:

- **cuRobo** installed in the Isaac Sim Python environment
- **CUDA** runtime + compatible NVIDIA drivers
- **PyTorch with CUDA support**

For lab staging runs, enforce the deterministic production path so SimReady uses
LLM-free physics with production gating.

```bash
RUN_STAGING_E2E=1 \
PIPELINE_ENV=production \
SIMREADY_PHYSICS_MODE=deterministic \
STAGING_SCENE_DIR=/mnt/gcs/scenes/<scene_id> \
/isaac-sim/python.sh -m pytest tests/test_pipeline_e2e_staging.py -v
```

You can also provide a data root + scene id instead of a full scene path:

```bash
RUN_STAGING_E2E=1 \
PIPELINE_ENV=production \
SIMREADY_PHYSICS_MODE=deterministic \
STAGING_DATA_ROOT=/mnt/gcs \
STAGING_SCENE_ID=<scene_id> \
/isaac-sim/python.sh -m pytest tests/test_pipeline_e2e_staging.py -v
```

## Production Mode

For production episode generation with full physics validation:

```bash
# Production mode enforces Isaac Sim usage and prevents mock fallback
PRODUCTION_MODE=true \
SCENE_ID=kitchen_scene_001 \
NUM_EPISODES=100 \
ROBOT_TYPE=franka \
MIN_QUALITY_SCORE=0.8 \
  /isaac-sim/python.sh episode-generation-job/generate_episodes.py
```

### Production Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `PRODUCTION_MODE` | Enforce Isaac Sim (no mock) | `false` | Yes (for prod) |
| `SCENE_ID` | Scene identifier | - | Yes |
| `NUM_EPISODES` | Episodes to generate | `10` | No |
| `ROBOT_TYPE` | Robot type (franka, ur5, etc.) | `franka` | No |
| `MIN_QUALITY_SCORE` | Min quality threshold | `0.7` | No |
| `ENABLE_VALIDATION` | Physics validation | `true` | No |
| `OUTPUT_DIR` | Episode output directory | `./output` | No |
| `PARQUET_VERIFICATION_MODE` | Parquet verification (`required`, `allow_fallback`, `disabled`) | `required` | No |

**Parquet dependency note:** Production exports run Parquet verification by default. Install `pyarrow` in production, or set `PARQUET_VERIFICATION_MODE=allow_fallback` with `pandas` + `fastparquet` installed if you need an alternate reader.

## Kubernetes Deployment (Production)

For large-scale episode generation in production:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: episode-generation
spec:
  template:
    spec:
      containers:
      - name: isaac-sim
        image: blueprint-isaac-sim:latest
        command: ["/isaac-sim/python.sh"]
        args:
          - "/workspace/BlueprintPipeline/episode-generation-job/generate_episodes.py"
        env:
          - name: PRODUCTION_MODE
            value: "true"
          - name: SCENE_ID
            value: "kitchen_001"
          - name: NUM_EPISODES
            value: "1000"
          - name: ROBOT_TYPE
            value: "franka"
        resources:
          requests:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
          limits:
            memory: "32Gi"
            cpu: "8"
            nvidia.com/gpu: "1"
        volumeMounts:
          - name: output
            mountPath: /output
          - name: scenes
            mountPath: /mnt/gcs
      volumes:
        - name: output
          persistentVolumeClaim:
            claimName: episode-output-pvc
        - name: scenes
          gcePersistentDisk:
            pdName: scene-assets
            fsType: ext4
      restartPolicy: OnFailure
```

## Troubleshooting

### GPU Not Detected

```bash
# Verify NVIDIA drivers
nvidia-smi

# Verify Docker can access GPU
docker run --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi

# If GPU not detected, install NVIDIA Container Toolkit:
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### Out of Memory

```bash
# Reduce batch size in episode generation
BATCH_SIZE=1 /isaac-sim/python.sh episode-generation-job/generate_episodes.py

# Or increase Docker memory limit
docker run --gpus all --memory=32g ...
```

### Display/X11 Errors

```bash
# On host
xhost +local:docker

# Verify DISPLAY is set in container
echo $DISPLAY

# If empty, set it
export DISPLAY=:0
```

### Isaac Sim Fails to Start

```bash
# Check Isaac Sim logs
cat ~/.nvidia-omniverse/logs/Kit/Isaac-Sim/*/kit_*.log

# Try headless mode (no GUI)
/isaac-sim/python.sh --headless episode-generation-job/generate_episodes.py

# Verify Vulkan support
vulkaninfo
```

## Performance Tuning

### GPU Optimization

```bash
# Enable TensorRT for faster rendering
export ENABLE_TENSORRT=1

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Multi-GPU (if available)
export CUDA_VISIBLE_DEVICES=0,1
```

### Memory Optimization

```bash
# Enable unified memory (slower but handles larger scenes)
export CUDA_UNIFIED_MEMORY=1

# Limit texture resolution
export MAX_TEXTURE_SIZE=2048
```

## Next Steps

- **Test Episode Generation**: Run a test generation with mock data
- **Quality Validation**: Review generated episodes for physics accuracy
- **Scale Up**: Deploy to Kubernetes for production workloads
- **Monitor**: Set up logging and metrics collection

## References

- [NVIDIA Isaac Sim Documentation](https://docs.omniverse.nvidia.com/isaacsim/)
- [Isaac Sim on NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim)
- [Omniverse Replicator](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html)
- [BlueprintPipeline Episode Generation](../episode-generation-job/README.md)
