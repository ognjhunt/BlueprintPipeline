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

## Binary Authorization

Production clusters must enable Binary Authorization. Configure it via tfvars:

- `enable_binary_authorization = true`
- Provide either:
  - A managed attestor key (`binary_authorization_attestor_public_key`), or
  - Existing attestors (`binary_authorization_attestors = ["projects/.../attestors/..."]`) with `create_binary_authorization_attestor = false`.
- If a project policy already exists, set `create_binary_authorization_policy = false` and import or manage it outside this module.

Terraform validates that Binary Authorization is enabled for `environment = "prod"`, and will fail `terraform plan` if it is disabled.
