# Data Retention Policy

This document defines the default retention windows and deletion triggers for BlueprintPipeline
artifacts. The retention configuration is enforced by a dedicated maintenance workflow that
runs the retention cleanup job on a schedule.

## Retention windows by artifact class

| Artifact class | Example paths | Default window | Purpose |
| --- | --- | --- | --- |
| **Inputs** | `scenes/{scene_id}/input/*` | 90 days | Preserve original uploads for auditability and reruns. |
| **Intermediate** | `scenes/{scene_id}/seg/*`, `scenes/{scene_id}/layout/*`, `scenes/{scene_id}/.checkpoints/*` | 30 days (via `PIPELINE_RETENTION_DAYS`) | Keep short-lived pipeline products and checkpoints needed for retries. |
| **Outputs** | `scenes/{scene_id}/assets/*`, `scenes/{scene_id}/usd/*`, `scenes/{scene_id}/replicator/*`, `scenes/{scene_id}/variation_assets/*`, `scenes/{scene_id}/isaac_lab/*` | 365 days | Retain deliverables for customers, exports, and downstream training. |
| **Episodes** | `scenes/{scene_id}/episodes/*` | 365 days | Retain training datasets; transition to colder storage after 365 days (see Terraform policy). |
| **Logs** | `scenes/{scene_id}/logs/*` | 180 days | Preserve operational logs for incident response and compliance. |

## Retention configuration

The maintenance job uses a pipeline-level retention setting to drive deletion decisions:

- `PIPELINE_RETENTION_DAYS` (default: **30**) controls the intermediate artifact window.
- Optional overrides are supported per class: `PIPELINE_INPUT_RETENTION_DAYS`,
  `PIPELINE_INTERMEDIATE_RETENTION_DAYS`, `PIPELINE_OUTPUT_RETENTION_DAYS`,
  and `PIPELINE_LOG_RETENTION_DAYS`.

These settings are read by `tools/checkpoint/retention_cleanup.py` and can be supplied via
Cloud Run job environment variables.

Infrastructure-backed defaults:

- The retention cleanup workflow and daily Cloud Scheduler trigger are provisioned in
  `infrastructure/terraform` (see `retention-cleanup.tf`).
- GCS lifecycle rules for `scenes/*` prefixes are managed in Terraform on the pipeline data
  bucket to align with the default retention windows. These rules act as a safety net
  alongside the workflow-driven cleanup job.
- Episode bundles are transitioned to the storage class defined by `episodes_retention_policy`
  instead of being deleted by lifecycle rules, so long-term training datasets can be retained.
- Firebase Storage lifecycle rules (if the Firebase bucket is managed in Terraform) align
  output (`datasets/`) and log (`logs/`) retention with this policy.

## Deletion triggers

Artifacts are deleted when **both** conditions are met:

1. The maintenance workflow executes the cleanup job (scheduled via Cloud Scheduler).
2. The artifact modification timestamp is older than the configured retention window for
   its class.

Additional deletion triggers:

- **Manual cleanup** during incident response or rollback procedures.
- **Customer or compliance requests** that require early data deletion.

## Audit logging

Every deletion (or dry-run deletion) is emitted as a structured log entry. These logs are
available in Cloud Logging and include:

- Artifact path and class
- Retention window applied
- Last-modified timestamp
- Deletion timestamp

Use these logs for auditing retention enforcement and validating compliance.

## Cleanup job and workflow

The retention cleanup runs as a dedicated maintenance workflow (`workflows/retention-cleanup.yaml`).
It triggers a Cloud Run job (`retention-cleanup-job`) that executes
`tools/checkpoint/retention_cleanup.py` against the canonical storage layout under
`gs://{BUCKET}/scenes/{scene_id}/`.

The workflow is scheduled daily via Cloud Scheduler when the Terraform configuration is
applied. Adjust the schedule and lifecycle rules in `infrastructure/terraform` to change
the default retention cadence or bucket policies.

The job emits structured audit logs for each deletion, which can be correlated with the
workflow execution ID in Cloud Logging.
