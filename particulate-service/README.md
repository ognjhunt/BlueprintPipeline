# Particulate Service

Feed-forward 3D object articulation service based on [Particulate](https://arxiv.org/abs/2512.11798) by Li et al.

## Overview

Particulate takes a 3D mesh as input and infers its articulated structure (parts, kinematic tree, joint types, axes, ranges) in a single feed-forward pass (~10 seconds).

**Key features:**

| Feature | Particulate |
|---------|-------------|
| Input | 3D mesh (GLB) |
| Speed | ~10 seconds |
| Memory | 16GB |
| Cold Start | 1-2 minutes |
| Strengths | Internal parts detection, batch processing |

## API

### POST / - Articulate Mesh

**Request:**
```json
{
  "glb_base64": "<base64-encoded GLB mesh>"
}
```

**Response:**
```json
{
  "mesh_base64": "<segmented GLB>",
  "urdf_base64": "<URDF with joints>",
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

Returns service status and readiness.

### GET /debug - Debug Info

Returns detailed service state and model validation info when debug access is explicitly enabled.
Debug is disabled by default and always blocked when `ENV=production`. For non-production
environments, set `DEBUG_MODE=1` and `DEBUG_TOKEN=<shared-secret>`, then send the token in an
`Authorization: Bearer <shared-secret>` header. Legacy `PARTICULATE_DEBUG` and
`PARTICULATE_DEBUG_TOKEN` variables are still honored.

## Deployment

See [BUILD_AND_DEPLOY.md](./BUILD_AND_DEPLOY.md) for detailed deployment instructions.

Quick start:
```bash
# Build
docker build -t particulate-service -f Dockerfile .

# Deploy to Cloud Run
gcloud run deploy particulate-service \
  --image <image-url> \
  --memory 16Gi --cpu 4 \
  --gpu 1 --gpu-type nvidia-l4 \
  --timeout 300 --concurrency 1
```

## Integration

Set this environment variable in interactive-job:

```bash
PARTICULATE_ENDPOINT=https://particulate-service-xxx.run.app
```

## References

- [Paper: arXiv:2512.11798](https://arxiv.org/abs/2512.11798)
- [GitHub: RuiningLi/particulate](https://github.com/RuiningLi/particulate)
- [HuggingFace Demo](https://huggingface.co/spaces/rayli/particulate/)
