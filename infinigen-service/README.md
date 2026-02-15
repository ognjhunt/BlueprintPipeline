# Infinigen Service

HTTP service that wraps [Infinigen](https://github.com/princeton-vl/infinigen) articulated asset generation and returns a **sim-ready** payload (URDF + referenced meshes) as a base64 zip.

This is intended to be used by `interactive-job` as the `infinigen` backend for supported categories.

## API

### `GET /`
Health check.

### `POST /`
Request:
```json
{
  "asset_name": "refrigerators",
  "seed": 1001,
  "collision": true,
  "export": "urdf"
}
```

Response:
```json
{
  "payload_zip_base64": "<base64 zip (URDF + meshes)>",
  "placeholder": false,
  "generator": "infinigen",
  "meta": {
    "asset_name": "refrigerators",
    "seed": 1001
  }
}
```

## Build

From repo root:
```bash
docker build -f infinigen-service/Dockerfile -t infinigen-service:latest .
```

## Run (local)

```bash
docker run --rm -p 8084:8080 \
  -e PORT=8080 \
  infinigen-service:latest
```

Then point `interactive-job` at `INFINIGEN_ENDPOINT=http://host.docker.internal:8084` (or the appropriate network alias).

