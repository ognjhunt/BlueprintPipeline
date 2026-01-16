# Rollback Procedures

Use this guide when a production pipeline change needs to be reverted quickly. Before rolling back, capture
current state (workflow run IDs, release tags, and infrastructure changes) so you can compare pre/post-fix
behavior.

## Job rollback

When a single job image or config change causes failures, roll back only that job while keeping the rest of the
pipeline intact.

1. Identify the last known-good image tag (for example, from the release registry or deployment history).
2. Redeploy the job with the previous tag.
3. Re-run the failed workflow or backfill the impacted scenes.

### Kubernetes deployment rollback (example)

```bash
# Inspect history for the deployment
kubectl rollout history deployment/scene-generation-job -n pipeline

# Roll back to the previous revision
kubectl rollout undo deployment/scene-generation-job -n pipeline

# Roll back to a specific revision (if needed)
kubectl rollout undo deployment/scene-generation-job -n pipeline --to-revision=12

# Verify status
kubectl rollout status deployment/scene-generation-job -n pipeline
```

### Cloud Run job rollback (example)

```bash
# Update a job to a prior image tag
gcloud run jobs update scene-generation-job \
  --image=gcr.io/<project>/scene-generation-job:<previous-tag> \
  --region=<region>
```

## Pipeline redeploy

If a shared dependency (workflow template, infrastructure module, or global config) caused widespread impact,
redeploy the full pipeline to the last known-good release.

1. Revert the workflow/template repo change (or checkout the previous release tag).
2. Redeploy the workflow and all jobs using the approved release pipeline.
3. Confirm the orchestration entrypoint is using the previous version.

### Terraform rollback (example)

```bash
# Inspect the state history (if remote state versioning is enabled)
terraform state list

# Re-apply the last known-good version from VCS
git checkout <previous-release-tag>
terraform init
terraform apply
```

### Terraform state rollback (example)

```bash
# Example: restore a previous state version (GCS backend)
gsutil ls gs://<state-bucket>/<prefix>

# Copy a known-good state version back to the active state file
gsutil cp gs://<state-bucket>/<prefix>/terraform.tfstate.<timestamp> \
  gs://<state-bucket>/<prefix>/terraform.tfstate

# Re-run Terraform to reconcile
terraform init
terraform apply
```

## Data cleanup

When rollback requires cleanup of partially processed data, remove the affected outputs and re-run the pipeline
from the last good checkpoint.

- Identify the impacted `SCENE_ID`s and their GCS prefixes.
- Remove intermediate artifacts that block reprocessing (e.g., `.regen3d_complete`, partial `scene.usda`).
- Re-run the workflow for just the affected scenes.

### Example cleanup commands

```bash
# Remove a partial completion marker
gsutil rm gs://<bucket>/<scene_prefix>/.regen3d_complete

# Remove generated artifacts for a specific scene
gsutil -m rm -r gs://<bucket>/<scene_prefix>/usd

# Re-run the workflow entrypoint (example)
gcloud workflows run blueprint-pipeline --data='{ "scene_id": "<scene_id>" }'
```
