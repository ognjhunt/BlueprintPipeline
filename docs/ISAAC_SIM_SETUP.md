# Isaac Sim Setup for BlueprintPipeline

This guide covers running the BlueprintPipeline with NVIDIA Isaac Sim for **real physics simulation** and **actual sensor data capture**.

## Why Isaac Sim?

Without Isaac Sim, the pipeline generates **mock data**:
- Random noise RGB images
- Heuristic-based validation (not physics-verified)
- No real sensor capture

With Isaac Sim, you get:
- **Real PhysX physics** simulation
- **Actual RGB/depth/segmentation** from Replicator
- **Physics-validated** trajectories
- **Production-quality** training data

## Prerequisites

### Hardware
- **NVIDIA GPU**: 8GB+ VRAM (T4, RTX 3080, A10, or better)
- **RAM**: 32GB+ recommended
- **Storage**: 50GB+ for Isaac Sim cache

### Software
- **Docker** with nvidia-docker runtime
- **NVIDIA Driver**: 525+
- **NGC Account** (for Isaac Sim container)

## Quick Start

### 1. Authenticate with NGC

```bash
# Get API key from https://ngc.nvidia.com/setup/api-key
docker login nvcr.io
# Username: $oauthtoken
# Password: <your-api-key>
```

### 2. Build the Container

```bash
cd BlueprintPipeline

# Build Isaac Sim container
docker build \
  -f episode-generation-job/Dockerfile.isaacsim \
  -t blueprint-episode-gen:isaacsim \
  .
```

### 3. Run Episode Generation

```bash
# Using the convenience script
./scripts/run-isaacsim-local.sh kitchen_001

# Or with Docker directly
docker run --rm --gpus all \
  --shm-size=16g \
  -e SCENE_ID=kitchen_001 \
  -v ./scenes:/mnt/local/scenes \
  -v ./output:/output \
  blueprint-episode-gen:isaacsim generate
```

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SCENE_ID` | (required) | Scene identifier |
| `ROBOT_TYPE` | `franka` | Robot: franka, ur10, fetch |
| `DATA_PACK_TIER` | `core` | Data tier: core, plus, full |
| `NUM_CAMERAS` | `1` | Number of cameras (1-4) |
| `EPISODES_PER_VARIATION` | `10` | Episodes per scene variation |
| `IMAGE_RESOLUTION` | `640,480` | Camera resolution |
| `MIN_QUALITY_SCORE` | `0.7` | Quality threshold (0-1) |
| `USE_LLM` | `true` | Use Gemini for task spec |
| `USE_CPGEN` | `true` | Use CP-Gen augmentation |

### Data Pack Tiers

| Tier | Includes | Use Case |
|------|----------|----------|
| **Core** | RGB + state + actions | Basic IL/BC training |
| **Plus** | Core + depth + segmentation + bboxes | Visual RL, perception |
| **Full** | Plus + poses + contacts + privileged | Sim2real, analysis |

## Docker Compose

For full pipeline orchestration:

```bash
# Set environment
export SCENE_ID=kitchen_001
export GEMINI_API_KEY=your-api-key

# Run episode generation
docker-compose -f docker-compose.isaacsim.yaml up episode-generation

# Run full pipeline
docker-compose -f docker-compose.isaacsim.yaml up pipeline

# Interactive shell
docker-compose -f docker-compose.isaacsim.yaml run --rm episode-generation shell
```

## Kubernetes / GKE Deployment

### Setup

```bash
# Create namespace and RBAC
kubectl apply -f k8s/namespace-setup.yaml

# Create secrets
kubectl create secret generic gcs-service-account \
  --from-file=key.json=./credentials/service-account.json \
  -n blueprint

kubectl create secret generic episode-gen-secrets \
  --from-literal=GEMINI_API_KEY=your-key \
  -n blueprint

# Create GPU node pool (if not exists)
gcloud container node-pools create gpu-pool \
  --cluster=your-cluster \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --num-nodes=1 \
  --enable-autoscaling --min-nodes=0 --max-nodes=10
```

### Run Job

```bash
# Edit job with your scene ID
sed -i 's/${SCENE_ID}/kitchen_001/g' k8s/episode-generation-job.yaml
sed -i 's/${PROJECT_ID}/your-project/g' k8s/episode-generation-job.yaml

# Apply
kubectl apply -f k8s/episode-generation-job.yaml

# Monitor
kubectl logs -f job/episode-generation -n blueprint
```

## Cloud Run (Limited Support)

Cloud Run doesn't support GPUs, so Isaac Sim cannot run there directly. Options:

1. **Vertex AI Custom Training**: Run as a training job with GPU
2. **GKE**: Use GPU node pools (recommended)
3. **Compute Engine**: Run on GPU VMs

## Troubleshooting

### Container won't start

```bash
# Check GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Check Isaac Sim health
docker run --rm --gpus all blueprint-episode-gen:isaacsim validate
```

### Out of Memory

```bash
# Increase shared memory
docker run --shm-size=32g ...

# Reduce resolution
-e IMAGE_RESOLUTION=320,240
```

### Isaac Sim import errors

Ensure you're using Isaac Sim's Python:
```bash
# Inside container
/isaac-sim/python.sh -c "import omni.isaac.core; print('OK')"
```

### Mock data still being generated

Check that `USE_MOCK_CAPTURE=false` and Isaac Sim is properly initialized:
```bash
docker run ... blueprint-episode-gen:isaacsim validate
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Docker Container                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                   Isaac Sim 4.2.0                          │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │ │
│  │  │    PhysX     │  │  Replicator  │  │     USD      │     │ │
│  │  │   Physics    │  │   Sensors    │  │    Stage     │     │ │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │ │
│  │         │                 │                 │              │ │
│  │         └─────────────────┼─────────────────┘              │ │
│  │                           │                                │ │
│  │  ┌────────────────────────▼────────────────────────────┐  │ │
│  │  │              Episode Generation Job                  │  │ │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │  │ │
│  │  │  │    Task     │  │   Motion    │  │   Sensor    │  │  │ │
│  │  │  │  Specifier  │  │   Planner   │  │   Capture   │  │  │ │
│  │  │  └─────────────┘  └─────────────┘  └─────────────┘  │  │ │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │  │ │
│  │  │  │   CP-Gen    │  │  Simulation │  │   LeRobot   │  │  │ │
│  │  │  │  Augmenter  │  │  Validator  │  │   Exporter  │  │  │ │
│  │  │  └─────────────┘  └─────────────┘  └─────────────┘  │  │ │
│  │  └─────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                        Output                               │ │
│  │   /output/scenes/{scene_id}/episodes/                       │ │
│  │   ├── lerobot/           # LeRobot dataset                  │ │
│  │   ├── manifests/         # Generation metadata              │ │
│  │   └── quality/           # Validation reports               │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Performance Tips

1. **Use SSD storage** for Isaac Sim cache
2. **Pre-warm** the container (first run is slow)
3. **Batch scenes** to amortize startup time
4. **Use T4 GPUs** for cost-effective runs
5. **Set `--shm-size=16g`** minimum

## Cost Estimates (GCP)

| Configuration | GPU | Cost/Hour | Episodes/Hour |
|---------------|-----|-----------|---------------|
| Development | T4 | ~$0.35 | ~50 |
| Production | A10 | ~$1.00 | ~150 |
| High-Volume | A100 | ~$3.00 | ~500 |

## Next Steps

After setting up Isaac Sim:

1. **Fix reward computation** - See `lerobot_exporter.py:137`
2. **Complete ground-truth generation** - See `lerobot_exporter.py:815`
3. **Implement UR10/Fetch IK** - See `trajectory_solver.py:267`

These are documented in `PIPELINE_GAP_ANALYSIS.md`.
