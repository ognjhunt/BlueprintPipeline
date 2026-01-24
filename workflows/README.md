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
| `training-pipeline.yaml` | Eventarc custom event | Event type `blueprintpipeline.episodes.imported` |
| `upsell-features-pipeline.yaml` | Manual only | Manual run with `.episodes_complete` payload |
| `usd-assembly-pipeline.yaml` | Eventarc (GCS finalized) | `scenes/*/assets/.regen3d_complete` |
| `variation-assets-pipeline.yaml` | Eventarc (GCS finalized) | `scenes/*/replicator/.replicator_complete` |

## Primary entrypoints
- YAML workflow definitions and the `setup-*.sh` scripts.

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
