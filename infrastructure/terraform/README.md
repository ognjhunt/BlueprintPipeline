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
  -var="project_id=YOUR_PROJECT_ID" \
  -var="tf_state_bucket=YOUR_TF_STATE_BUCKET" \
  -var="tf_state_prefix=YOUR_ENVIRONMENT/terraform/state"

terraform apply \
  -var="project_id=YOUR_PROJECT_ID" \
  -var="tf_state_bucket=YOUR_TF_STATE_BUCKET" \
  -var="tf_state_prefix=YOUR_ENVIRONMENT/terraform/state"
```
