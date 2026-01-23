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

## Key environment variables
- Environment variables used by trigger setup scripts and workflow runtime configuration.
- `PRIMARY_WORKFLOW_REGION`: primary region for Cloud Run job invocations. Defaults to `us-central1`.
- `SECONDARY_WORKFLOW_REGION`: secondary region used when primary health checks fail. Defaults to `us-east1`.
- `WORKFLOW_REGION`: legacy override for workflows that do not yet use primary/secondary routing.
- `PRIMARY_BUCKET`: default bucket for manual or scheduler-driven workflows (replaces hardcoded bucket names).
- `FIREBASE_STORAGE_BUCKET`: required in production workflow environments that enable Firebase uploads for Genie Sim export/import.

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
