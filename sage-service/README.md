# sage-service

## Purpose
HTTP wrapper for Stage-1 SAGE backend calls from BlueprintPipeline.

Expected endpoint from Stage-1:
- `POST /v1/refine`

## API
- `GET /healthz`
- `POST /v1/refine`

Request fields (minimum):
- `scene_id`
- `prompt`
- `quality_tier`
- `seed`
- `constraints`
- `base_scene` (optional, used for hybrid/serial refine)

Response contract:
- `{"package": {...}}` or `{"objects": [...]}`

## Modes
- `SAGE_SERVICE_MODE=internal`
  - Uses in-repo SAGE refinement strategy implementation (default).
- `SAGE_SERVICE_MODE=command`
  - Runs `SAGE_COMMAND` and passes request JSON on stdin.
  - Command must return JSON on stdout.
- `SAGE_SERVICE_MODE=http_forward`
  - Forwards request to `SAGE_UPSTREAM_URL`.

Common env:
- `PORT` (default `8082` when running directly)
- `SAGE_SERVICE_TIMEOUT_SECONDS` (default `1800`)
- `CORS_ALLOWED_ORIGINS` (optional)

## Local run
```bash
pip install -r sage-service/requirements.txt
python sage-service/sage_service.py
```

## Cloud Run build/deploy
```bash
gcloud builds submit --tag us-central1-docker.pkg.dev/<PROJECT>/<REPO>/sage-service:latest -f sage-service/Dockerfile .

gcloud run deploy sage-service \
  --image us-central1-docker.pkg.dev/<PROJECT>/<REPO>/sage-service:latest \
  --region us-central1 \
  --allow-unauthenticated
```

Then set pipeline env:
- `SAGE_SERVER_URL=https://<service-url>/v1/refine`
