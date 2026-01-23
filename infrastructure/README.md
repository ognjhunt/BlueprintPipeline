# infrastructure

## Purpose / scope
Infrastructure-as-code and operational automation for the Blueprint Pipeline.

## Primary entrypoints
- `terraform/` infrastructure definitions.
- `scripts/` helper scripts for infrastructure tasks.
- `monitoring/` monitoring configuration.

## Required inputs / outputs
- **Inputs:** cloud credentials, Terraform variables, and environment configuration.
- **Outputs:** provisioned infrastructure, monitoring dashboards, and logs.

## Key environment variables
- Cloud provider credentials and Terraform variables used by scripts and Terraform.

## How to run locally
- Run Terraform and scripts directly from their respective directories after exporting the required credentials and variables.

## Updating GKE master authorized networks
- Update `infrastructure/terraform/terraform.tfvars` (or your environment-specific tfvars) to maintain the `master_authorized_networks` list.
- Add approved CIDR blocks for CI/CD runners and developer VPNs using the `{ cidr_block, display_name }` entries.
- Re-run `terraform plan` and `terraform apply` from `infrastructure/terraform/` to apply changes.
