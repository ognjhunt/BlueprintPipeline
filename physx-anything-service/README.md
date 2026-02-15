# PhysX-Anything Service

GPU HTTP service that wraps [PhysX-Anything](https://github.com/VictorTao1998/PhysX-Anything) and returns a **sim-ready** payload (URDF + referenced meshes) as a base64 zip.

This is intended to be used by `interactive-job` as the `physx_anything` backend.

## API

### `GET /`
Health check.

### `POST /`
Request:
```json
{
  "image_base64": "<base64 PNG/JPG bytes>",
  "seed": 1001,
  "remove_bg": false,
  "voxel_define": 32,
  "fixed_base": 0,
  "deformable": 0
}
```

Response:
```json
{
  "payload_zip_base64": "<base64 zip (URDF + meshes)>",
  "placeholder": false,
  "generator": "physx-anything",
  "articulation": {
    "joint_count": 2,
    "is_articulated": true
  }
}
```

## Build

From repo root:
```bash
docker build -f physx-anything-service/Dockerfile -t physx-anything-service:latest .
```

This Dockerfile runs `python download.py` at build time, so a fresh container starts without downloading checkpoints.

## Run (local)

```bash
docker run --rm --gpus all -p 8083:8080 \
  -e PORT=8080 \
  physx-anything-service:latest
```

Then point `interactive-job` at `PHYSX_ANYTHING_ENDPOINT=http://host.docker.internal:8083` (or the appropriate network alias).

