# Terraform Infrastructure

## Remote state initialization

This configuration uses a GCS backend for remote state. Provide the backend
configuration at initialization time so you can reuse the same code across
multiple environments without hardcoding project-specific values.

### Option 1: Inline backend config

```bash
terraform init \
  -backend-config="bucket=YOUR_TF_STATE_BUCKET" \
  -backend-config="prefix=YOUR_ENVIRONMENT/terraform/state"
```

### Option 2: Backend config file per environment

Create a backend config file (for example, `backend.dev.hcl`):

```hcl
bucket = "YOUR_TF_STATE_BUCKET"
prefix = "dev/terraform/state"
```

Then initialize with:

```bash
terraform init -backend-config="backend.dev.hcl"
```

## Planning and applying

```bash
terraform plan \
  -var="project_id=YOUR_PROJECT_ID"

terraform apply \
  -var="project_id=YOUR_PROJECT_ID"
```

> **Note**
> Several resources explicitly depend on `google_project_service.apis` to avoid a first-time bootstrap race where
> Terraform attempts to create IAM, GKE, Pub/Sub, or monitoring resources before the corresponding APIs finish
> enabling. This keeps initial `terraform apply` runs reliable in new projects.

## Required inputs

| Variable | Description |
| --- | --- |
| `project_id` | GCP project ID used for all resources. |

## Monitoring resources

Terraform now provisions the dashboards, alert policies, and log-based metrics checked by the workflow monitoring
gate. Override names in non-prod environments using the variables below or a `*.tfvars` file.

| Variable | Description | Default |
| --- | --- | --- |
| `monitoring_dashboard_overview_name` | Display name for the overview dashboard. | `BlueprintPipeline - Overview` |
| `monitoring_dashboard_gpu_name` | Display name for the GPU metrics dashboard. | `BlueprintPipeline - GPU Metrics` |
| `monitoring_alert_workflow_timeout_name` | Display name for the workflow timeout alert policy. | `[Blueprint] Workflow Job Timeout Detected` |
| `monitoring_alert_workflow_retry_spike_name` | Display name for the workflow retry spike alert policy. | `[Blueprint] Workflow Job Retry Spike` |
| `monitoring_notification_channel_ids` | Notification channel IDs for monitoring alert policies. | `[]` |

Example override (`terraform.tfvars`):

```hcl
monitoring_dashboard_overview_name       = "BlueprintPipeline (Staging) - Overview"
monitoring_dashboard_gpu_name            = "BlueprintPipeline (Staging) - GPU Metrics"
monitoring_alert_workflow_timeout_name   = "[Blueprint][Staging] Workflow Job Timeout Detected"
monitoring_alert_workflow_retry_spike_name = "[Blueprint][Staging] Workflow Job Retry Spike"
monitoring_notification_channel_ids      = ["projects/YOUR_PROJECT_ID/notificationChannels/1234567890"]
```

## Binary Authorization

Production clusters must enable Binary Authorization. Configure it via tfvars:

- `enable_binary_authorization = true`
- Provide either:
  - A managed attestor key (`binary_authorization_attestor_public_key`), or
  - Existing attestors (`binary_authorization_attestors = ["projects/.../attestors/..."]`) with `create_binary_authorization_attestor = false`.
- If a project policy already exists, set `create_binary_authorization_policy = false` and import or manage it outside this module.

Terraform validates that Binary Authorization is enabled for `environment = "prod"`, and will fail `terraform plan` if it is disabled.
