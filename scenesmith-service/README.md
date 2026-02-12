# scenesmith-service

## Purpose
HTTP wrapper for Stage-1 SceneSmith backend calls from BlueprintPipeline.

Expected endpoint from Stage-1:
- `POST /v1/generate`

## API
- `GET /healthz`
- `POST /v1/generate`

Request fields (minimum):
- `scene_id`
- `prompt`
- `quality_tier`
- `seed`
- `constraints`

Response contract:
- `{"package": {...}}` or `{"objects": [...]}`

## Modes
- `SCENESMITH_SERVICE_MODE=internal`
  - Uses in-repo SceneSmith strategy implementation (default).
- `SCENESMITH_SERVICE_MODE=command`
  - Runs `SCENESMITH_COMMAND` and passes request JSON on stdin.
  - Command must return JSON on stdout.
- `SCENESMITH_SERVICE_MODE=http_forward`
  - Forwards request to `SCENESMITH_UPSTREAM_URL`.
- `SCENESMITH_SERVICE_MODE=paper_stack` (or `paper`)
  - Runs the official SceneSmith stack through the command bridge at
    `/Users/nijelhunt_1/workspace/BlueprintPipeline/scenesmith-service/scenesmith_paper_command.py`.
  - Requires these env vars:
    - `SCENESMITH_PAPER_REPO_DIR` (path to cloned official `nepfaff/scenesmith` repo)
    - `SCENESMITH_PAPER_PYTHON_BIN` (python inside that repo's env, optional; default `python3`)
  - Optional controls:
    - `SCENESMITH_PAPER_BACKEND` (`openai|gemini|anthropic`, default `openai`)
    - `SCENESMITH_PAPER_MODEL` (override backend model)
    - `SCENESMITH_PAPER_TIMEOUT_SECONDS` (default `5400`)
    - `SCENESMITH_PAPER_RUN_ROOT` (default `/tmp/scenesmith-paper-runs`)
    - `SCENESMITH_PAPER_KEEP_RUN_DIR` (`true|false`, default `false`)
    - `SCENESMITH_PAPER_SPLIT_WALLS`, `SCENESMITH_PAPER_VALIDATE_PHYSICS`,
      `SCENESMITH_PAPER_GENERATE_OVERVIEW_IMAGES`

Common env:
- `PORT` (default `8081` when running directly)
- `SCENESMITH_SERVICE_TIMEOUT_SECONDS` (default `3600`)
- `CORS_ALLOWED_ORIGINS` (optional)

## Local run
```bash
pip install -r scenesmith-service/requirements.txt
python scenesmith-service/scenesmith_service.py
```

## Cloud Run build/deploy
```bash
gcloud builds submit --tag us-central1-docker.pkg.dev/<PROJECT>/<REPO>/scenesmith-service:latest -f scenesmith-service/Dockerfile .

gcloud run deploy scenesmith-service \
  --image us-central1-docker.pkg.dev/<PROJECT>/<REPO>/scenesmith-service:latest \
  --region us-central1 \
  --allow-unauthenticated
```

Then set pipeline env:
- `SCENESMITH_SERVER_URL=https://<service-url>/v1/generate`
