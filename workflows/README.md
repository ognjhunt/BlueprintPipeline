# workflows

## Purpose / scope
Workflow definitions and trigger setup scripts for pipeline orchestration.

## Index
- `*-pipeline.yaml` workflow definitions (e.g., `arena-export-pipeline.yaml`, `usd-assembly-pipeline.yaml`).
- `*-poller.yaml` poller workflows (e.g., `genie-sim-import-poller.yaml`).
- `setup-*.sh` trigger setup scripts.
- `TIMEOUT_AND_RETRY_POLICY.md` workflow policy reference.

## Workflow trigger map
| Workflow | Trigger source | Marker file or scheduler |
| --- | --- | --- |
| `asset-embedding-pipeline.yaml` | Eventarc (GCS finalized) | `automation/asset_embedding/queue/*.json` |
| `arena-export-pipeline.yaml` | Eventarc (GCS finalized) | `scenes/*/usd/.usd_complete`, `scenes/*/isaac_lab/.isaac_lab_complete`, `scenes/*/geniesim/.geniesim_complete` |
| `dream2flow-preparation-pipeline.yaml` | Eventarc (GCS finalized) | `scenes/*/assets/.regen3d_complete` (disabled unless enabled) |
| `dwm-preparation-pipeline.yaml` | Eventarc (GCS finalized) | `scenes/*/assets/.regen3d_complete` (disabled unless enabled) |
| `episode-generation-pipeline.yaml` | Eventarc (GCS finalized) | `scenes/*/usd/.usd_complete` |
| `genie-sim-export-pipeline.yaml` | Eventarc (GCS finalized) | `scenes/*/variation_assets/.variation_pipeline_complete` |
| `genie-sim-import-pipeline.yaml` | Eventarc custom event / manual | Event type `manual.geniesim.job.completed` or direct workflow payload |
| `genie-sim-import-poller.yaml` | Cloud Scheduler | Every 5–10 minutes; scans `scenes/*/geniesim/job.json` |
| `interactive-pipeline.yaml` | Eventarc (GCS finalized) | `scenes/*/assets/scene_assets.json` |
| `objects-pipeline.yaml` | Eventarc (GCS finalized) | `scenes/*/layout/scene_layout.json` |
| `regen3d-pipeline.yaml` | Eventarc (GCS finalized) | `scenes/*/assets/.assets_ready` |
| `retention-cleanup.yaml` | Cloud Scheduler | Daily retention cleanup (managed in `infrastructure/terraform`) |
| `scale-pipeline.yaml` | Eventarc (GCS finalized) | `scenes/*/layout/scene_layout.json` |
| `scene-batch.yaml` | Manual only | Provide a manifest object with a scene list |
| `scene-generation-pipeline.yaml` | Cloud Scheduler / manual | Scheduler (disabled by default) or manual run |
| `source-orchestrator.yaml` | Eventarc (GCS finalized) | `scenes/*/prompts/scene_request.json` |
| `text-autonomy-daily.yaml` | Cloud Scheduler | Daily autonomous text request emission + downstream completion watch |
| `training-pipeline.yaml` | Eventarc custom event | Event type `blueprintpipeline.episodes.imported` |
| `upsell-features-pipeline.yaml` | Manual only | Manual run with `.episodes_complete` payload |
| `usd-assembly-pipeline.yaml` | Eventarc (GCS finalized) | `scenes/*/assets/.regen3d_complete` |
| `variation-assets-pipeline.yaml` | Eventarc (GCS finalized) | `scenes/*/replicator/.replicator_complete` |

## Primary entrypoints
- YAML workflow definitions and the `setup-*.sh` scripts.

## Source Orchestrator (text-first)
The `source-orchestrator.yaml` workflow is the text-first entry point for scene generation.
It replaces Stage 1 reconstruction with LLM/template-based scene generation from a text prompt,
then feeds the result into the same Stage 2-5 pipeline used by the image path.

- **Text path**: `text-scene-gen-job` → `text-scene-adapter-job` → Stage 2-5
- **Text runtime modes**: `TEXT_GEN_RUNTIME=vm|cloudrun|runpod` (`vm` default; `runpod` uses on-demand L40S Stage 1)
- **Stage 1 Cloud Run overrides**: `TEXT_SCENE_GEN_JOB_NAME` and `TEXT_SCENE_ADAPTER_JOB_NAME` allow non-default job names
- **LLM-first Stage 1**: `TEXT_GEN_USE_LLM=true` default with deterministic fallback metadata and retry controls
- **Stage 5 strictness**: `ARENA_EXPORT_REQUIRED=true` default (text completion requires Arena success)
- **Live backend services**: optional SceneSmith/SAGE endpoints via `SCENESMITH_SERVER_URL` and `SAGE_SERVER_URL`
- **Image path (compat mode)**:
  - `IMAGE_PATH_MODE=orchestrator` delegates to `image-to-scene-orchestrator`
  - `IMAGE_PATH_MODE=legacy_chain` delegates to `image-to-scene-pipeline` and waits for `.geniesim_complete`
- **Auto path**: tries text first, falls back to image if text fails and an image is present
- **Multi-seed**: `seed_count > 1` fans out to child scene IDs (`{scene_id}-s001`, `-s002`, etc.)
- **Child request path**: fanout requests are written to `scenes/<child_scene_id>/internal/scene_request.generated.json` (non-trigger path)
- **Dedupe lock**: per-generation lock at `scenes/<scene_id>/locks/source-orchestrator-<generation>.lock` prevents concurrent duplicate executions
- **Fallback**: fail-fast — if any seed fails, the entire request falls back to image (does not retry remaining seeds)

Service setup guide:
- `/Users/nijelhunt_1/workspace/BlueprintPipeline/docs/text_backend_services.md`

## Stage 4 Runtime Contract
- Required Cloud Run job: `genie-sim-export-job`.
- `genie-sim-submit-job` and `genie-sim-gpu-job` are not required as Cloud Run jobs in the current architecture.
- Submission/execution is handled by the existing Cloud Build + GKE/local runtime path in `genie-sim-export-pipeline.yaml`.

## Source Request Contract (Text-First)
`source-orchestrator.yaml` expects prompt requests at:

`gs://<bucket>/scenes/<scene_id>/prompts/scene_request.json`

Contract schema:

- `fixtures/contracts/scene_request_v1.schema.json`

Core fields:

- `schema_version` (`v1`)
- `scene_id`
- `source_mode` (`text` | `image` | `auto`)
- `text_backend` (`internal` | `scenesmith` | `sage` | `hybrid_serial`) optional; defaults to workflow/backend config
- `prompt` (required for `text`/`auto`)
- `quality_tier` (`standard` | `premium`)
- `seed_count` (>= 1)
- `image.gcs_uri` (required for `image` mode)
- `provider_policy` (`openrouter_qwen_primary` | `openai_primary`)
- `fallback.allow_image_fallback` (default `true`)

**Note:** Dream2Flow and DWM workflows are experimental and remain disabled by
default unless explicitly enabled (e.g., `ENABLE_DREAM2FLOW=true`,
`ENABLE_DWM=true`, or a shared experimental flag).

## Required inputs / outputs
- **Inputs:** workflow parameters, referenced container images, and pipeline configuration.
- **Outputs:** triggered pipeline runs, workflow logs, and artifacts.

## Genie Sim idempotency markers
Genie Sim workflows record idempotency markers in GCS to prevent duplicate submissions/imports.

### Export submission marker
- **Path:** `scenes/{scene_id}/geniesim/idempotency/{hash}.json`
- **Hash inputs:** `scene_id`, `task_config`, `robot_types`, `episodes_per_task`, `quality_thresholds`
- **Body (example):**
```json
{
  "scene_id": "scene_001",
  "status": "submitted",
  "job_id": "job_abc123",
  "timestamp": "2024-01-01T00:00:00Z",
  "hash": "4b825dc642cb6eb9a060e54bf8d69288",
  "inputs": {
    "task_config": {},
    "robot_types": ["franka"],
    "episodes_per_task": "10",
    "quality_thresholds": {}
  }
}
```

### Import marker
- **Path:** `scenes/{scene_id}/geniesim/idempotency/import/{job_id}.json`
- **Body (example):**
```json
{
  "scene_id": "scene_001",
  "job_id": "job_abc123",
  "status": "completed",
  "timestamp": "2024-01-01T00:30:00Z"
}
```

## Key environment variables
- Environment variables used by trigger setup scripts and workflow runtime configuration.
- `PRIMARY_WORKFLOW_REGION`: primary region for Cloud Run job invocations. Defaults to `us-central1`.
- `SECONDARY_WORKFLOW_REGION`: secondary region used when primary health checks fail. Defaults to `us-east1`.
- `WORKFLOW_REGION`: legacy override for workflows that do not yet use primary/secondary routing.
- `PRIMARY_BUCKET`: default bucket for manual or scheduler-driven workflows (replaces hardcoded bucket names).
- `DEFAULT_SOURCE_MODE`: source-orchestrator default request mode (`text`, `image`, `auto`). Defaults to `text`.
- `TEXT_BACKEND_DEFAULT`: default Stage 1 text backend when request omits `text_backend`. Defaults to `sage`.
- `TEXT_BACKEND_ALLOWLIST`: allowed Stage 1 text backends. Defaults to `internal,scenesmith,sage,hybrid_serial`.
- `TEXT_GEN_RUNTIME`: source-orchestrator runtime profile hint for text Stage 1 (`vm`, `cloudrun`, `runpod`). Defaults to `vm`.
- `TEXT_SCENE_GEN_JOB_NAME`: Stage 1 Cloud Run job name override for text scene generation. Defaults to `text-scene-gen-job`.
- `TEXT_SCENE_ADAPTER_JOB_NAME`: Stage 1 Cloud Run job name override for text scene adaptation. Defaults to `text-scene-adapter-job`.
- `TEXT_GEN_STANDARD_PROFILE`: profile label injected into text-scene-gen-job for `quality_tier=standard`.
- `TEXT_GEN_PREMIUM_PROFILE`: profile label injected into text-scene-gen-job for `quality_tier=premium`.
- `TEXT_GEN_USE_LLM`: enables LLM-first scene planning in Stage 1. Defaults to `true`.
- `TEXT_GEN_LLM_MAX_ATTEMPTS`: max LLM planning retry rounds in Stage 1. Defaults to `3`.
- `TEXT_GEN_LLM_RETRY_BACKOFF_SECONDS`: Stage 1 LLM retry backoff base seconds. Defaults to `2`.
- `TEXT_OPENROUTER_API_KEY`: OpenRouter API key override for `openrouter_qwen_primary` (falls back to `OPENROUTER_API_KEY`).
- `TEXT_OPENROUTER_BASE_URL`: OpenRouter-compatible base URL for Stage 1 OpenAI-compatible calls. Defaults to `https://openrouter.ai/api/v1`.
- `TEXT_OPENROUTER_MODEL_CHAIN`: ordered OpenRouter model attempts (comma-list or JSON list). Defaults to `qwen/qwen3.5-397b-a17b,moonshotai/kimi-k2.5`.
- `TEXT_OPENROUTER_INCLUDE_LEGACY_FALLBACK`: append legacy Stage 1 provider fallbacks (`openai`, `anthropic`) after OpenRouter attempts. Defaults to `true`.
- `TEXT_ASSET_RETRIEVAL_ENABLED`: text adapter retrieval toggle before placeholder fallback. Defaults to `true`.
- `TEXT_ASSET_LIBRARY_PREFIXES`: comma-separated library prefixes under bucket mount for retrieval (e.g. `scenes,asset-library`). Defaults to `scenes`.
- `TEXT_ASSET_LIBRARY_MAX_FILES`: scan cap for retrieval index build in text adapter. Defaults to `2500`.
- `TEXT_ASSET_LIBRARY_MIN_SCORE`: minimum token-match score for selecting retrieved assets. Defaults to `0.25`.
- `TEXT_ASSET_RETRIEVAL_MODE`: retrieval rollout mode (`lexical_primary`, `ann_shadow`, `ann_primary`). Defaults to `ann_shadow`.
- `TEXT_ASSET_ANN_ENABLED`: enable ANN semantic retrieval path. Defaults to `true`.
- `TEXT_ASSET_ANN_TOP_K`: ANN candidate query size before rerank. Defaults to `40`.
- `TEXT_ASSET_ANN_MIN_SCORE`: minimum ANN semantic similarity accepted. Defaults to `0.28`.
- `TEXT_ASSET_ANN_MAX_RERANK`: max ANN candidates reranked in adapter. Defaults to `20`.
- `TEXT_ASSET_ANN_NAMESPACE`: vector namespace/collection for asset embeddings. Defaults to `assets-v1`.
- `TEXT_ASSET_LEXICAL_FALLBACK_ENABLED`: allow lexical fallback when ANN misses/fails. Defaults to `true`.
- `TEXT_ASSET_ROLLOUT_STATE_PREFIX`: rollout state object prefix. Defaults to `automation/asset_retrieval_rollout`.
- `TEXT_ASSET_ROLLOUT_MIN_DECISIONS`: min ANN decisions in a rollout window. Defaults to `500`.
- `TEXT_ASSET_ROLLOUT_MIN_HIT_RATE`: ANN hit-rate threshold for promotion. Defaults to `0.95`.
- `TEXT_ASSET_ROLLOUT_MAX_ERROR_RATE`: ANN error-rate threshold for promotion. Defaults to `0.01`.
- `TEXT_ASSET_ROLLOUT_MAX_P95_MS`: ANN p95 latency threshold for promotion. Defaults to `400`.
- `TEXT_ASSET_CATALOG_ENABLED`: publish text adapter asset/scene metadata to Firestore catalog. Defaults to `true`.
- `TEXT_ASSET_EMBEDDING_QUEUE_PREFIX`: embedding queue prefix. Defaults to `automation/asset_embedding/queue`.
- `TEXT_ASSET_EMBEDDING_PROCESSED_PREFIX`: embedding success result prefix. Defaults to `automation/asset_embedding/processed`.
- `TEXT_ASSET_EMBEDDING_FAILED_PREFIX`: embedding failure result prefix. Defaults to `automation/asset_embedding/failed`.
- `TEXT_ASSET_EMBEDDING_MODEL`: embedding model used by retrieval/indexing. Defaults to `text-embedding-3-small`.
- `TEXT_ASSET_REPLICATION_ENABLED`: enqueue async replication requests. Defaults to `true`.
- `TEXT_ASSET_REPLICATION_QUEUE_PREFIX`: replication queue prefix. Defaults to `automation/asset_replication/queue`.
- `TEXT_ASSET_REPLICATION_TARGET`: replication target label. Defaults to `backblaze_b2`.
- `TEXT_ASSET_REPLICATION_TARGET_PREFIX`: replication object-key prefix. Defaults to `assets`.
- `TEXT_ASSET_GENERATION_ENABLED`: enable generation provider fallback when retrieval misses. Defaults to `true`.
- `TEXT_ASSET_GENERATION_PROVIDER`: text asset generation provider (`sam3d` default).
- `TEXT_ASSET_GENERATION_PROVIDER_CHAIN`: ordered fallback providers (default `sam3d,hunyuan3d`).
- `TEXT_ASSET_GENERATED_CACHE_ENABLED`: cache generated assets for future retrieval. Defaults to `true`.
- `TEXT_ASSET_GENERATED_CACHE_PREFIX`: cache prefix for generated assets. Defaults to `asset-library/generated-text`.
- `TEXT_SAM3D_API_HOST`: SAM3D provider base URL.
- `TEXT_SAM3D_TEXT_ENDPOINTS`: endpoint candidates for SAM3D text-to-3D. Defaults to `/openapi/v1/text-to-3d,/v1/text-to-3d`.
- `TEXT_SAM3D_TIMEOUT_SECONDS`: SAM3D task timeout seconds. Defaults to `1800`.
- `TEXT_SAM3D_POLL_SECONDS`: SAM3D poll interval seconds. Defaults to `10`.
- `TEXT_HUNYUAN_API_HOST`: Hunyuan fallback provider base URL.
- `TEXT_HUNYUAN_TEXT_ENDPOINTS`: endpoint candidates for Hunyuan text-to-3D. Defaults to `/openapi/v1/text-to-3d,/v1/text-to-3d`.
- `TEXT_HUNYUAN_TIMEOUT_SECONDS`: Hunyuan task timeout seconds. Defaults to `1800`.
- `TEXT_HUNYUAN_POLL_SECONDS`: Hunyuan poll interval seconds. Defaults to `10`.
- `TEXT_GEN_MAX_SEEDS`: max allowed `seed_count` for `scene_request.json`. Defaults to `16`.
- `TEXT_GEN_ENABLE_IMAGE_FALLBACK`: allow `auto`/`text` fallback to image path when text source fails. Defaults to `true`.
- `ARENA_EXPORT_REQUIRED`: enforce Stage 5 arena success before source completion. Defaults to `true`.
- `IMAGE_PATH_MODE`: image compatibility mode (`orchestrator` or `legacy_chain`). Defaults to `orchestrator`.
- `IMAGE_ORCHESTRATOR_WORKFLOW_NAME`: workflow used when `IMAGE_PATH_MODE=orchestrator`. Defaults to `image-to-scene-orchestrator`.
- `IMAGE_LEGACY_WORKFLOW_NAME`: workflow used when `IMAGE_PATH_MODE=legacy_chain`. Defaults to `image-to-scene-pipeline`.
- `IMAGE_LEGACY_CHAIN_WAIT_SECONDS`: timeout while waiting for `scenes/<scene_id>/geniesim/.geniesim_complete` in legacy-chain mode. Defaults to `7200`.
- `TEXT_GEN_VM_NAME`: VM name used when `TEXT_GEN_RUNTIME=vm`. Defaults to `isaac-sim-ubuntu`.
- `TEXT_GEN_VM_ZONE`: VM zone used when `TEXT_GEN_RUNTIME=vm`. Defaults to `us-east1-c`.
- `TEXT_GEN_VM_REPO_DIR`: repository directory on the VM for text Stage 1 scripts. Defaults to `~/BlueprintPipeline`.
- `TEXT_GEN_VM_TIMEOUT_SECONDS`: VM text-stage Cloud Build polling timeout. Defaults to `2400`.
- `RUNPOD_API_KEY`: required when `TEXT_GEN_RUNTIME=runpod`.
- `RUNPOD_GPU_TYPE`: RunPod GPU type (default `NVIDIA L40S`) for Stage 1 runpod runtime.
- `RUNPOD_IMAGE`: RunPod container image for Stage 1 runpod runtime.
- `RUNPOD_REPO_DIR`: repository directory inside RunPod pod. Defaults to `/workspace/BlueprintPipeline`.
- `RUNPOD_SCENESMITH_REPO_DIR`: official SceneSmith checkout path in RunPod pod. Defaults to `/workspace/scenesmith`.
- `RUNPOD_BOOTSTRAP_COMMAND`: optional command executed on RunPod before repo-dir validation (use this to clone/install on fresh pods).
- `RUNPOD_TERMINATE_ON_EXIT`: terminate pod after Stage 1 completes (`true` default).
- `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `GEMINI_API_KEY`, `HF_TOKEN`, `GITHUB_TOKEN`: optional secrets forwarded to RunPod for Stage 1 backends/bootstrap.
- `SCENESMITH_RUNTIME_MODE`: runtime selector for SceneSmith backend execution metadata (`cloudrun` default).
- `SCENESMITH_SERVER_URL`: optional SceneSmith server URL used by Stage 1 generation.
- `SCENESMITH_TIMEOUT_SECONDS`: timeout hint for SceneSmith backend calls (`1800` default).
- `SAGE_RUNTIME_MODE`: runtime selector for SAGE backend execution metadata (`cloudrun` default).
- `SAGE_SERVER_URL`: optional SAGE server URL used by Stage 1 generation.
- `SAGE_TIMEOUT_SECONDS`: timeout hint for SAGE backend calls (`900` default).
- `TEXT_SAGE_ACTION_DEMO_ENABLED`: emit non-blocking Franka action-demo artifacts under `textgen/sage_actions` (`false` default).
- `VECTOR_STORE_PROVIDER`: vector database backend (`vertex` default).
- `VECTOR_STORE_PROJECT_ID`: vector backend project.
- `VECTOR_STORE_LOCATION`: vector backend location.
- `VECTOR_STORE_NAMESPACE`: vector namespace/collection (`assets-v1` default).
- `VECTOR_STORE_DIMENSION`: vector dimensionality (`1536` default).
- `VERTEX_INDEX_ENDPOINT`: Vertex Matching Engine endpoint resource.
- `VERTEX_DEPLOYED_INDEX_ID`: Vertex deployed index ID.
- `B2_S3_ENDPOINT`: Backblaze B2 S3 endpoint for async replication worker (job-level env).
- `B2_BUCKET`: Backblaze B2 target bucket (job-level env).
- `B2_REGION`: Backblaze B2 region (default `us-west-000`, job-level env).
- `B2_KEY_ID_SECRET`: Secret Manager secret name bound to `B2_KEY_ID` in replication job.
- `B2_APPLICATION_KEY_SECRET`: Secret Manager secret name bound to `B2_APPLICATION_KEY` in replication job.
- `TEXT_AUTONOMY_STATE_PREFIX`: daily text autonomy state root. Defaults to `automation/text_daily`.
- `TEXT_AUTONOMY_TIMEZONE`: scheduler/workflow timezone hint for daily runs. Defaults to `America/New_York`.
- `TEXT_DAILY_QUOTA`: number of scenes emitted per daily run. Defaults to `1`.
- `TEXT_DAILY_PAUSE_AFTER_CONSEC_FAILS`: auto-pause threshold for autonomy workflow. Defaults to `3`.
- `ENABLE_VLM_QUALITY_AUDIT`: enables VLM episode audit in Genie Sim export/import path. Defaults to `1` in `genie-sim-export-pipeline.yaml`.
- `FIREBASE_STORAGE_BUCKET`: required in production workflow environments that enable Firebase uploads for Genie Sim export/import.
- `GENIESIM_CIRCUIT_BREAKER_THRESHOLD`: maximum consecutive failures before Genie Sim export/import workflows short-circuit. Defaults to `3`.

## Genie Sim circuit breaker
The Genie Sim export/import workflows maintain a per-scene circuit breaker file at:

```
scenes/<scene_id>/geniesim/.circuit_breaker.json
```

The file is updated after failures and reset on successful runs to prevent repeated job retries when a scene is unhealthy.
The structure is:

```json
{
  "scene_id": "scene_123",
  "failure_count": 2,
  "threshold": 3,
  "status": "open",
  "updated_at": "2024-05-01T12:34:56Z",
  "last_failure_at": "2024-05-01T12:34:56Z",
  "last_failure_reason": "geniesim_export_failed",
  "last_success_at": "2024-04-30T10:12:00Z"
}
```

When `failure_count` meets or exceeds `GENIESIM_CIRCUIT_BREAKER_THRESHOLD`, the workflows log the circuit breaker
state and exit early until a successful run resets the counter.

## Region routing logic
Selected workflows perform regional health checks before invoking Cloud Run jobs:

1. Try the primary region (`PRIMARY_WORKFLOW_REGION`) by calling `run.jobs.get`.
2. If the primary region is unavailable, try the secondary region (`SECONDARY_WORKFLOW_REGION`).
3. Route job executions to the first healthy region and log any regional failures.

This keeps new workloads running during regional outages while preserving idempotency markers in GCS.

## Policy compliance
- Standard retry/backoff and timeout defaults live in `TIMEOUT_AND_RETRY_POLICY.md` and `policy_configs/adaptive_timeouts.yaml`.
- If a workflow needs a non-default timeout, document the override in the workflow header and ensure the policy doc stays in sync.

## How to run locally
- Run trigger setup scripts directly (e.g., `./setup-all-triggers.sh`) after exporting the required credentials.

## Text autonomy daily workflow usage
`text-autonomy-daily.yaml` runs fully autonomous daily text generation by invoking
`text-request-emitter-job`, then waiting for source-orchestrator scene completion.

Setup:
```bash
cd workflows
TEXT_DAILY_QUOTA=1 \
TEXT_AUTONOMY_TIMEZONE=America/New_York \
bash setup-text-autonomy-scheduler.sh <project_id> <bucket> <region>
```

State objects:
- `automation/text_daily/state.json`
- `automation/text_daily/.paused`
- `automation/text_daily/runs/<YYYY-MM-DD>/emitted_requests.json`
- `automation/text_daily/runs/<YYYY-MM-DD>/run_summary.json`

## Asset replication queue workflow usage
`asset-replication-pipeline.yaml` processes async replication queue objects and
invokes `asset-replication-job` to mirror assets from GCS to Backblaze B2.

Setup:
```bash
cd workflows
bash setup-asset-replication-trigger.sh <project_id> <bucket> <region>
```

Optional secure credential binding during setup:
```bash
cd workflows
bash setup-backblaze-secrets.sh <project_id>

B2_S3_ENDPOINT=https://s3.us-west-000.backblazeb2.com \
B2_BUCKET=<b2-bucket> \
B2_KEY_ID_SECRET=b2-key-id \
B2_APPLICATION_KEY_SECRET=b2-application-key \
bash setup-asset-replication-trigger.sh <project_id> <bucket> <region>
```

The setup script keeps credentials out of workflow env vars and binds secrets directly
to the `asset-replication-job` Cloud Run job.

Queue objects:
- `automation/asset_replication/queue/*.json`

Result objects:
- `automation/asset_replication/processed/*.json`
- `automation/asset_replication/failed/*.json`

## Asset embedding queue workflow usage
`asset-embedding-pipeline.yaml` processes async embedding queue objects and
invokes `asset-embedding-job` to upsert ANN vectors used by Stage 1 retrieval.

Setup:
```bash
cd workflows
OPENAI_API_KEY_SECRET=<secret-name> \
VECTOR_STORE_PROVIDER=vertex \
VECTOR_STORE_PROJECT_ID=<project_id> \
VECTOR_STORE_LOCATION=<region> \
VERTEX_INDEX_ENDPOINT=<vertex-endpoint-resource> \
VERTEX_DEPLOYED_INDEX_ID=<deployed-index-id> \
bash setup-asset-embedding-trigger.sh <project_id> <bucket> <region>
```

Queue objects:
- `automation/asset_embedding/queue/*.json`

Result objects:
- `automation/asset_embedding/processed/*.json`
- `automation/asset_embedding/failed/*.json`

Catalog backfill helper:
```bash
BUCKET=<bucket> \
TEXT_ASSET_EMBEDDING_QUEUE_PREFIX=automation/asset_embedding/queue \
TEXT_ASSET_EMBEDDING_BACKFILL_STATE_PREFIX=automation/asset_embedding/backfill \
python tools/asset_catalog/backfill_embeddings.py
```

## Scene batch workflow usage
The `scene-batch.yaml` workflow runs the batch pipeline across a list of scene IDs
provided via a JSON manifest in GCS. The manifest can be either a JSON list of
scene IDs or an object with a `scenes` array.

Example manifest:
```json
{
  "scenes": ["scene_001", "scene_002", "scene_003"]
}
```

Trigger the workflow:
```bash
gcloud workflows run scene-batch \
  --location=us-central1 \
  --data='{"bucket":"your-bucket","manifest_object":"scene-batches/manifest.json","max_concurrent":10,"retry_attempts":2}'
```

Parallelism controls:
- `max_concurrent` controls the number of scenes processed in parallel.
- `retry_attempts`, `retry_delay`, and `rate_limit` tune retry and throttling behavior.

Outputs:
- `reports_dir/batch_report.json` now includes `scene_reports` entries with a per-scene
  `quality_gate_report` path for monitoring.
## Canary staging validation runbook
Use the staging script below before production releases to validate canary routing
and rollback handling for `genie-sim-export-pipeline.yaml`.

### Prerequisites
- `gcloud` authenticated to the staging project.
- `jq` installed (used by the script to parse execution IDs).
- Two scene IDs with completed variation assets:
  - `CANARY_SCENE_ID` (should route to canary image tag).
  - `STABLE_SCENE_ID` (should route to stable image tag).

### Optional rollback marker
To validate rollback behavior, create the rollback marker object for the canary
scene before running the rollback validation:

```bash
gsutil cp /dev/null gs://$BUCKET/scenes/$CANARY_SCENE_ID/geniesim/.canary_rollback
```

### Run the staging validation
```bash
export PROJECT_ID="your-project"
export BUCKET="your-staging-bucket"
export CANARY_SCENE_ID="scene_canary_001"
export STABLE_SCENE_ID="scene_stable_001"
export CANARY_IMAGE_TAG="isaacsim-canary"
export CANARY_PERCENT="5"
export CANARY_RELEASE_CHANNEL="canary"
export CANARY_ROLLBACK_MARKER="scenes/$CANARY_SCENE_ID/geniesim/.canary_rollback"

./run-canary-staging.sh
```

### Validation checklist
The script writes logs/artifacts to `workflows/artifacts/canary-validation/<timestamp>/`:
- `stable-workflow-logs.json` should include the routing log with
  `canary_enabled=false` and `image_tag` set to the stable image tag.
- `canary-workflow-logs.json` should include the routing log with
  `canary_enabled=true` and `image_tag` set to the canary image tag.
- `rollback-workflow-logs.json` (if run) should include
  `Rollback marker present ... Skipping Genie Sim export.`

Use `jq` or `rg` against the JSON logs to confirm the routing lines.

### Canary export artifact validation
After the canary execution completes, `run-canary-staging.sh` downloads Genie Sim
export outputs from `gs://$BUCKET/scenes/$CANARY_SCENE_ID/geniesim/` and validates:
- Required artifacts exist: `scene_graph.json`, `asset_index.json`, `task_config.json`.
- JSON schema checks using fixtures in `fixtures/contracts/`:
  - `scene_graph.schema.json`
  - `asset_index.schema.json`
  - `task_config.schema.json`
- Cross-file consistency via `tools/validation/geniesim_export.validate_export_consistency`.

Validation results are written to:
- `${ARTIFACT_ROOT}/canary-validation.json` (structured success report).
- `${ARTIFACT_ROOT}/canary-validation.log` (schema or consistency errors).

If validation fails, the script exits non-zero and writes a failure marker to:
`gs://$BUCKET/scenes/$CANARY_SCENE_ID/canary-validation/` using the shared failure
marker format. Review the validation log and the failure marker payload before
promoting a canary image.
