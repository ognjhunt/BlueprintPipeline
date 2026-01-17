# Particulate Service Build and Deploy Guide

## Overview

Particulate is a feed-forward 3D articulation model that takes a mesh as input and outputs:
- Part segmentation
- Kinematic tree (parent-child relationships)
- Joint types (revolute/prismatic)
- Joint axes and motion ranges
- URDF for simulation

**Key features:**
- **Fast**: ~10 seconds per object
- **Efficient memory**: 16GB RAM
- **Quick cold starts**: ~1-2 min
- **Geometry-based**: Trained on mesh geometry features

**Use cases:**
- When you have a 3D mesh (from 3D-RE-GEN) and need articulation
- Batch processing of many objects
- Cost-sensitive deployments

## Quick Start

### Prerequisites
- Docker with buildx
- Google Cloud SDK configured

### 1. Build the Docker Image

```bash
cd /path/to/BlueprintPipeline

docker build \
  --platform linux/amd64 \
  -t us-central1-docker.pkg.dev/blueprint-8c1ca/blueprint/particulate-service:latest \
  -f particulate-service/Dockerfile \
  particulate-service/
```

**Note**: Build takes ~15-20 minutes.

### 2. Push to GCP Artifact Registry

```bash
docker push us-central1-docker.pkg.dev/blueprint-8c1ca/blueprint/particulate-service:latest
```

### 3. Deploy to Cloud Run

```bash
gcloud run deploy particulate-service \
  --image us-central1-docker.pkg.dev/blueprint-8c1ca/blueprint/particulate-service:latest \
  --project blueprint-8c1ca \
  --region us-central1 \
  --platform managed \
  --memory 16Gi \
  --cpu 4 \
  --gpu 1 \
  --gpu-type nvidia-l4 \
  --timeout 300 \
  --concurrency 1 \
  --min-instances 0 \
  --max-instances 4 \
  --port 8080 \
  --no-cpu-throttling \
  --set-env-vars "PARTICULATE_DEBUG=1,PARTICULATE_DEBUG_TOKEN=<shared-secret>" \
  --allow-unauthenticated
```

**Critical settings:**
- `--concurrency 1`: GPU can only handle one request at a time
- `--memory 16Gi`: Efficient memory usage
- `--timeout 300`: 5 minutes (inference is fast, ~10s)
- `--min-instances 0`: Scales to zero (cold starts are fast)
- `--max-instances 4`: Can handle more parallel requests

### 4. Test the Service

```bash
# Health check
curl https://particulate-service-744608654760.us-central1.run.app/

# Debug info (requires explicit debug token)
curl -H "Authorization: Bearer <shared-secret>" \
  https://particulate-service-744608654760.us-central1.run.app/debug

# Test with a mesh
python test_particulate_service.py
```

## API Reference

### POST / - Process Mesh

**Request:**
```json
{
  "glb_base64": "<base64-encoded GLB mesh>"
}
```

**Response:**
```json
{
  "mesh_base64": "<base64-encoded segmented GLB>",
  "urdf_base64": "<base64-encoded URDF>",
  "placeholder": false,
  "generator": "particulate",
  "articulation": {
    "joint_count": 2,
    "part_count": 3,
    "is_articulated": true
  }
}
```

### GET / - Health Check

**Response (healthy):**
```json
{
  "status": "ok",
  "ready": true,
  "service": "particulate"
}
```

### GET /debug - Debug Info

Returns detailed service state including model validation when debug access is enabled.
Set `PARTICULATE_DEBUG=1` and `PARTICULATE_DEBUG_TOKEN=<shared-secret>` and send the
token in an `Authorization: Bearer <shared-secret>` header to receive a `200` response.
Missing or invalid credentials return `403`.

## Integration with Interactive Job

Set environment variables in the interactive-job to use Particulate:

```bash
# Use Particulate as the articulation backend
ARTICULATION_BACKEND=particulate

# Particulate service endpoint
PARTICULATE_ENDPOINT=https://particulate-service-xxx.run.app
```

### Provisioning the Endpoint for Workflows

For production workflows, make sure the interactive job receives the
`PARTICULATE_ENDPOINT` value so articulation detection can run. After the Cloud
Run deployment, capture the service URL and wire it into the pipeline:

```bash
# Example: set the endpoint on the Cloud Run job used by workflows
gcloud run jobs update interactive-job \
  --region us-central1 \
  --set-env-vars "PARTICULATE_ENDPOINT=https://particulate-service-xxx.run.app,DISALLOW_PLACEHOLDER_URDF=true"
```

Alternatively, export `PARTICULATE_ENDPOINT` in the workflow environment or CI
runtime so `workflows/interactive-pipeline.yaml` can pass it through to the job.
In production, set `DISALLOW_PLACEHOLDER_URDF=true` to fail fast if Particulate
returns placeholder responses.

## Particulate Features

| Feature | Particulate |
|---------|-------------|
| **Input** | 3D mesh (GLB/OBJ) |
| **Output** | Parts + URDF |
| **Speed** | ~10 seconds |
| **Memory** | 16GB |
| **Cold Start** | 1-2 minutes |
| **Model Size** | ~100MB |
| **Strengths** | Internal parts detection, fast inference |

### When to use Particulate:
- You have a 3D mesh from 3D-RE-GEN
- Fast turnaround is important
- Batch processing many objects
- Budget-conscious deployments

## Troubleshooting

### Model not found
```
Particulate directory not found at /opt/particulate
```
- Check that the git clone succeeded during build
- Verify the PARTICULATE_ROOT environment variable

### CUDA not available
```
CUDA test failed
```
- Ensure GPU is attached: `--gpu 1 --gpu-type nvidia-l4`
- Check NVIDIA driver logs in Cloud Run

### Inference timeout
```
Pipeline timed out (120s)
```
- Should not happen (inference takes ~10s)
- Check for OOM issues with large meshes
- Reduce mesh complexity before sending

## Cost Comparison

**Particulate** (per 1000 objects):
- GPU time: ~3 hours (at ~10s each)
- Cost: ~$2.10 (L4 at $0.70/hr)

## References

- [Particulate Paper (arXiv:2512.11798)](https://arxiv.org/abs/2512.11798)
- [Particulate GitHub](https://github.com/RuiningLi/particulate)
- [HuggingFace Demo](https://huggingface.co/spaces/rayli/particulate/)
- [Cloud Run GPU docs](https://cloud.google.com/run/docs/configuring/services/gpu)
