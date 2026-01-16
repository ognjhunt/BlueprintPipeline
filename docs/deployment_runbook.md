# Deployment Runbook

This runbook captures the standard production deployment flow for the BlueprintPipeline on Google Cloud. It covers
infrastructure apply, secrets, Cloud Run / GKE deployments, and workflow activation. For rollback guidance, see
[docs/rollback.md](rollback.md).

## Scope

Use this runbook when:
- Standing up a new environment.
- Releasing new job images or workflow updates.
- Rotating secrets or updating infrastructure modules.

## Prerequisites

- Access to the target GCP project (Owner/Editor + Workflows/Cloud Run/GKE permissions).
- `gcloud`, `terraform`, and `kubectl` installed locally.
- Container image tags for each job (builds already pushed to Artifact Registry/GCR).
- The desired environment variables or secrets for workloads.

## Preflight validation

Run these checks **before** applying infrastructure or deploying workloads:

```bash
# Confirm project + auth

gcloud auth list

gcloud config get-value project

# Confirm required APIs

gcloud services list --enabled | rg -e run.googleapis.com -e workflows.googleapis.com -e eventarc.googleapis.com

# Validate Terraform modules

terraform -chdir=infrastructure init

terraform -chdir=infrastructure fmt -check

terraform -chdir=infrastructure validate

terraform -chdir=infrastructure plan -out=tfplan

# (Optional) confirm cluster access if deploying to GKE

kubectl version --client

kubectl config current-context
```

If any step fails, fix the issue before proceeding. For deployment failure recovery, see
[docs/rollback.md](rollback.md).

## Step 1: Apply Terraform

Apply the infrastructure changes using the approved environment workspace.

```bash
# Example: use a dedicated workspace

terraform -chdir=infrastructure workspace select <env> || \
  terraform -chdir=infrastructure workspace new <env>

terraform -chdir=infrastructure apply tfplan
```

Confirm that:
- Cloud Run jobs/services are present.
- Artifact Registry repos exist.
- Workflows and Eventarc triggers are created.

## Step 2: Create or rotate secrets

Store runtime secrets in Secret Manager. Use the names expected by Terraform/workflow configuration.

```bash
# Create a new secret (example)

gcloud secrets create PIPELINE_API_KEY --replication-policy="automatic"

echo -n "<value>" | gcloud secrets versions add PIPELINE_API_KEY --data-file=-

# Update an existing secret (adds a new version)

echo -n "<new-value>" | gcloud secrets versions add PIPELINE_API_KEY --data-file=-
```

Ensure IAM bindings allow the runtime service accounts to access the secret versions.

## Step 3: Deploy Cloud Run jobs

Update each pipeline job to point at the desired image tag. Use the same region as the infrastructure.

```bash
# Example job update

gcloud run jobs update scene-generation-job \
  --image=us-docker.pkg.dev/<project>/<repo>/scene-generation-job:<tag> \
  --region=<region>

# Validate that jobs are ready

gcloud run jobs describe scene-generation-job --region=<region>
```

Repeat for each job defined in the pipeline (regen3d, simready, usd-assembly, replicator, isaac-lab, etc.).

## Step 4: Deploy GKE workloads (if applicable)

If any pipeline steps run in GKE (e.g., GPU workloads), update the deployment manifests and apply:

```bash
# Set cluster context

gcloud container clusters get-credentials <cluster-name> --region <region>

# Update the image on the deployment

kubectl -n pipeline set image deployment/scene-generation-job \
  scene-generation-job=us-docker.pkg.dev/<project>/<repo>/scene-generation-job:<tag>

# Monitor rollout status

kubectl -n pipeline rollout status deployment/scene-generation-job
```

## Step 5: Activate workflows and triggers

Deploy or update the Cloud Workflow definition and confirm triggers are active.

```bash
# Deploy workflow

gcloud workflows deploy blueprint-pipeline \
  --source=workflows/usd-assembly-pipeline.yaml \
  --location=<region>

# Verify Eventarc trigger

gcloud eventarc triggers list --location=<region>
```

Optionally run a canary workflow with a known scene ID:

```bash

gcloud workflows run blueprint-pipeline --data='{ "scene_id": "<scene_id>" }'
```

## Post-deploy validation

- Confirm recent Cloud Run job executions succeed.
- Validate that Eventarc triggers fire on `.regen3d_complete` markers.
- Spot check GCS outputs for a recent scene.

If anything fails, reference the rollback procedures in [docs/rollback.md](rollback.md).
