# Text Backend Services (SceneSmith + SAGE)

This document describes how to run SceneSmith and SAGE as live HTTP services for Stage 1 text generation.

## What Stage 1 expects

Stage 1 calls two optional HTTP endpoints:
- `SCENESMITH_SERVER_URL` -> `POST /v1/generate`
- `SAGE_SERVER_URL` -> `POST /v1/refine`

If endpoints are set and reachable, Stage 1 uses them.
If unreachable, Stage 1 falls back to internal generation/refinement logic.

## Service wrappers in this repo

- `/Users/nijelhunt_1/workspace/BlueprintPipeline/scenesmith-service/scenesmith_service.py`
- `/Users/nijelhunt_1/workspace/BlueprintPipeline/sage-service/sage_service.py`

Both expose:
- `GET /healthz`
- `POST` contract endpoint

Both support runtime modes:
- `internal`: run in-repo implementation (default)
- `command`: execute external backend command via stdin/stdout JSON
- `http_forward`: forward to upstream URL
- `paper_stack` (SceneSmith only): run official SceneSmith stack via command bridge

## Quick start on VM (recommended first)

### 1) Start both services

```bash
cd /Users/nijelhunt_1/workspace/BlueprintPipeline
./scripts/setup_text_backend_services.sh
export PYTHON_BIN=/Users/nijelhunt_1/workspace/BlueprintPipeline/.venv-text-backends/bin/python
./scripts/start_text_backend_services.sh start
./scripts/start_text_backend_services.sh status
```

Defaults:
- SceneSmith on `http://127.0.0.1:8081/v1/generate`
- SAGE on `http://127.0.0.1:8082/v1/refine`

### 2) Configure pipeline env

Set in workflow deploy environment:

```bash
SCENESMITH_SERVER_URL=http://127.0.0.1:8081/v1/generate
SAGE_SERVER_URL=http://127.0.0.1:8082/v1/refine
TEXT_BACKEND_DEFAULT=scenesmith
TEXT_GEN_RUNTIME=vm
```

Then redeploy source orchestrator trigger/workflow:

```bash
cd /Users/nijelhunt_1/workspace/BlueprintPipeline/workflows
bash setup-source-orchestrator-trigger.sh <project_id> <bucket> <region>
```

### 3) Validate services

```bash
curl -s http://127.0.0.1:8081/healthz | jq .
curl -s http://127.0.0.1:8082/healthz | jq .
```

## Cloud Run deployment

Use helper script:

```bash
cd /Users/nijelhunt_1/workspace/BlueprintPipeline
./scripts/deploy_text_backend_services.sh <project_id> <region> <artifact_repo>
```

Then set:
- `SCENESMITH_SERVER_URL=https://<scenesmith-service-url>/v1/generate`
- `SAGE_SERVER_URL=https://<sage-service-url>/v1/refine`

Redeploy source orchestrator after env updates.

## Using external upstream implementations

To use actual upstream SceneSmith/SAGE servers:

### SceneSmith wrapper
- `SCENESMITH_SERVICE_MODE=http_forward`
- `SCENESMITH_UPSTREAM_URL=<your-real-scenesmith-endpoint>`

Or command bridge:
- `SCENESMITH_SERVICE_MODE=command`
- `SCENESMITH_COMMAND="<cmd that reads JSON stdin and writes JSON stdout>"`

Or official SceneSmith paper stack (still through same local endpoint):
- `SCENESMITH_SERVICE_MODE=paper_stack`
- `SCENESMITH_PAPER_REPO_DIR=<path to nepfaff/scenesmith checkout>`
- `SCENESMITH_PAPER_PYTHON_BIN=<python in official scenesmith venv>`
- Optional:
  - `SCENESMITH_PAPER_BACKEND=openai|gemini|anthropic` (default `openai`)
  - `SCENESMITH_PAPER_MODEL=<backend-specific model id>`
  - `SCENESMITH_PAPER_TIMEOUT_SECONDS=5400`
  - `SCENESMITH_PAPER_KEEP_RUN_DIR=false`

Example VM launch for paper stack:

```bash
cd /Users/nijelhunt_1/workspace/BlueprintPipeline
export SCENESMITH_SERVICE_MODE=paper_stack
export SCENESMITH_PAPER_REPO_DIR=/home/nijelhunt1/scenesmith
export SCENESMITH_PAPER_PYTHON_BIN=/home/nijelhunt1/scenesmith/.venv/bin/python
export SCENESMITH_PAPER_BACKEND=openai
export SCENESMITH_PAPER_MODEL=gpt-4o
./scripts/start_text_backend_services.sh restart
```

### SAGE wrapper
- `SAGE_SERVICE_MODE=http_forward`
- `SAGE_UPSTREAM_URL=<your-real-sage-endpoint>`

Or command bridge:
- `SAGE_SERVICE_MODE=command`
- `SAGE_COMMAND="<cmd that reads JSON stdin and writes JSON stdout>"`

## Backend selection at request time

In `scene_request.json`:
- `text_backend: "scenesmith"`
- `text_backend: "sage"`
- `text_backend: "hybrid_serial"` (SceneSmith then SAGE)

Example:

```json
{
  "schema_version": "v1",
  "scene_id": "scene_demo_001",
  "source_mode": "text",
  "text_backend": "scenesmith",
  "prompt": "A cluttered kitchen where a robot moves a bowl to a shelf",
  "quality_tier": "standard",
  "seed_count": 1,
  "provider_policy": "openai_primary"
}
```
