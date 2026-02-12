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

## Step 2a: Configure cost tracking pricing

Production cost tracking **requires** real pricing inputs. Store the pricing files in a ConfigMap
or Secret Manager and wire the env vars into production jobs so `CostTracker` loads them.

### Required environment variables
- `COST_TRACKING_PRICING_PATH` or `COST_TRACKING_PRICING_JSON` (pricing overrides file/payload)
- `GENIESIM_JOB_COST`
- `GENIESIM_EPISODE_COST`
- `GENIESIM_GPU_RATE_TABLE_PATH` or `GENIESIM_GPU_RATE_TABLE` (GPU-hour pricing; optional if
  `GENIESIM_GPU_HOURLY_RATE` is set)

### Example pricing files

`cost-tracking-pricing.json` (per-job + per-episode overrides):
```json
{
  "gemini_input_per_1k": 0.00125,
  "gemini_output_per_1k": 0.005,
  "cloud_run_vcpu_second": 0.000024,
  "cloud_run_memory_gb_second": 0.0000025,
  "cloud_build_minute": 0.003,
  "gcs_storage_gb_month": 0.02,
  "gcs_operation_class_a": 0.0000005,
  "gcs_operation_class_b": 0.00000004,
  "geniesim_job": 1.25,
  "geniesim_episode": 0.03
}
```

`geniesim-gpu-rate-table.json` (GPU-hour pricing):
```json
{
  "default": {
    "g5.xlarge": 1.006,
    "g5.2xlarge": 1.212,
    "g5.12xlarge": 4.384,
    "a2-highgpu-1g": 1.685
  }
}
```

### Kubernetes example

Apply the bundled ConfigMaps and ensure production jobs mount them:

```bash
kubectl apply -f k8s/cost-tracking-pricing.yaml
```

Confirm the runtime env includes:

```bash
kubectl -n blueprint get configmap cost-tracking-pricing -o yaml
```

### Automated rotation (Cloud Scheduler + Cloud Run job)

The pipeline supports automated secret rotation via a Cloud Run job and a Cloud Scheduler trigger managed by
Terraform. The job generates new secret versions at a fixed cadence and relies on workloads to use `latest`
secret versions at runtime (Cloud Run/GKE workloads fetch `latest` by default). Restart long-lived services
after rotation if they cache secrets in memory.

**Configure rotation in Terraform**

1. Build/push the rotation job image from `infrastructure/secret-rotation/`.
2. Update Terraform variables:
   - `secret_rotation_job_image`
   - `secret_rotation_secret_ids`
   - `secret_rotation_schedule` / `secret_rotation_time_zone`
3. Apply Terraform to deploy the scheduler + job.

**Validation**

```bash
gcloud run jobs describe secret-rotation-job --region=<region>
gcloud scheduler jobs describe secret-rotation-schedule --location=<region>
```

**Rollback**

1. Pause the scheduler:

```bash
gcloud scheduler jobs pause secret-rotation-schedule --location=<region>
```

2. Pin the desired secret version (if a rollback is required):

```bash
gcloud secrets versions list <SECRET_ID>
gcloud secrets versions access <VERSION> --secret=<SECRET_ID> > /tmp/secret-value
echo -n \"$(cat /tmp/secret-value)\" | gcloud secrets versions add <SECRET_ID> --data-file=-
```

3. Restart Cloud Run services/jobs or GKE workloads if they cache secrets in memory.

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

For the text-first source orchestrator, deploy triggers with an explicit image compatibility mode:

```bash
cd workflows

# Recommended: orchestrator topology (avoids duplicate marker-chain triggers)
IMAGE_PATH_MODE=orchestrator \
TEXT_GEN_RUNTIME=vm \
bash setup-all-triggers.sh <project_id> <bucket> <region> orchestrator

# Legacy compatibility mode (keeps marker-chain trigger topology)
IMAGE_PATH_MODE=legacy_chain \
TEXT_GEN_RUNTIME=cloudrun \
bash setup-all-triggers.sh <project_id> <bucket> <region> legacy_chain
```

Key source-orchestrator env vars to set at deploy time:
- `DEFAULT_SOURCE_MODE`
- `TEXT_BACKEND_DEFAULT`
- `TEXT_BACKEND_ALLOWLIST`
- `TEXT_GEN_RUNTIME`
- `TEXT_GEN_STANDARD_PROFILE`
- `TEXT_GEN_PREMIUM_PROFILE`
- `TEXT_GEN_USE_LLM`
- `TEXT_GEN_LLM_MAX_ATTEMPTS`
- `TEXT_GEN_LLM_RETRY_BACKOFF_SECONDS`
- `TEXT_ASSET_RETRIEVAL_ENABLED`
- `TEXT_ASSET_LIBRARY_PREFIXES`
- `TEXT_ASSET_LIBRARY_MAX_FILES`
- `TEXT_ASSET_LIBRARY_MIN_SCORE`
- `TEXT_ASSET_RETRIEVAL_MODE`
- `TEXT_ASSET_ANN_ENABLED`
- `TEXT_ASSET_ANN_TOP_K`
- `TEXT_ASSET_ANN_MIN_SCORE`
- `TEXT_ASSET_ANN_MAX_RERANK`
- `TEXT_ASSET_ANN_NAMESPACE`
- `TEXT_ASSET_LEXICAL_FALLBACK_ENABLED`
- `TEXT_ASSET_ROLLOUT_STATE_PREFIX`
- `TEXT_ASSET_ROLLOUT_MIN_DECISIONS`
- `TEXT_ASSET_ROLLOUT_MIN_HIT_RATE`
- `TEXT_ASSET_ROLLOUT_MAX_ERROR_RATE`
- `TEXT_ASSET_ROLLOUT_MAX_P95_MS`
- `TEXT_ASSET_CATALOG_ENABLED`
- `TEXT_ASSET_EMBEDDING_QUEUE_PREFIX`
- `TEXT_ASSET_EMBEDDING_PROCESSED_PREFIX`
- `TEXT_ASSET_EMBEDDING_FAILED_PREFIX`
- `TEXT_ASSET_EMBEDDING_MODEL`
- `TEXT_ASSET_REPLICATION_ENABLED`
- `TEXT_ASSET_REPLICATION_QUEUE_PREFIX`
- `TEXT_ASSET_REPLICATION_TARGET`
- `TEXT_ASSET_REPLICATION_TARGET_PREFIX`
- `TEXT_ASSET_GENERATION_ENABLED`
- `TEXT_ASSET_GENERATION_PROVIDER`
- `TEXT_ASSET_GENERATION_PROVIDER_CHAIN`
- `TEXT_ASSET_GENERATED_CACHE_ENABLED`
- `TEXT_ASSET_GENERATED_CACHE_PREFIX`
- `TEXT_SAM3D_API_HOST`
- `TEXT_SAM3D_TEXT_ENDPOINTS`
- `TEXT_SAM3D_TIMEOUT_SECONDS`
- `TEXT_SAM3D_POLL_SECONDS`
- `TEXT_HUNYUAN_API_HOST`
- `TEXT_HUNYUAN_TEXT_ENDPOINTS`
- `TEXT_HUNYUAN_TIMEOUT_SECONDS`
- `TEXT_HUNYUAN_POLL_SECONDS`
- `TEXT_GEN_MAX_SEEDS`
- `TEXT_GEN_ENABLE_IMAGE_FALLBACK`
- `ARENA_EXPORT_REQUIRED`
- `IMAGE_PATH_MODE`
- `IMAGE_ORCHESTRATOR_WORKFLOW_NAME`
- `IMAGE_LEGACY_WORKFLOW_NAME`
- `IMAGE_LEGACY_CHAIN_WAIT_SECONDS`
- `TEXT_GEN_VM_NAME`
- `TEXT_GEN_VM_ZONE`
- `TEXT_GEN_VM_REPO_DIR`
- `TEXT_GEN_VM_TIMEOUT_SECONDS`
- `SCENESMITH_RUNTIME_MODE`
- `SCENESMITH_SERVER_URL`
- `SCENESMITH_TIMEOUT_SECONDS`
- `SAGE_RUNTIME_MODE`
- `SAGE_SERVER_URL`
- `SAGE_TIMEOUT_SECONDS`
- `TEXT_SAGE_ACTION_DEMO_ENABLED`
- `VECTOR_STORE_PROVIDER`
- `VECTOR_STORE_PROJECT_ID`
- `VECTOR_STORE_LOCATION`
- `VECTOR_STORE_NAMESPACE`
- `VECTOR_STORE_DIMENSION`
- `VERTEX_INDEX_ENDPOINT`
- `VERTEX_DEPLOYED_INDEX_ID`

Optional live text backend service setup (SceneSmith/SAGE):

```bash
cd /Users/nijelhunt_1/workspace/BlueprintPipeline

# Local VM wrappers
./scripts/setup_text_backend_services.sh
export PYTHON_BIN=/Users/nijelhunt_1/workspace/BlueprintPipeline/.venv-text-backends/bin/python
./scripts/start_text_backend_services.sh start

# Or deploy wrappers to Cloud Run
./scripts/deploy_text_backend_services.sh <project_id> <region> <artifact_repo>
```

Then set:
- `SCENESMITH_SERVER_URL` to `/v1/generate`
- `SAGE_SERVER_URL` to `/v1/refine`

Full guide:
- `/Users/nijelhunt_1/workspace/BlueprintPipeline/docs/text_backend_services.md`

For async Backblaze replication, deploy the replication queue workflow trigger:

```bash
cd workflows
bash setup-asset-replication-trigger.sh <project_id> <bucket> <region>
```

Replication workflow env vars:
- `ASSET_REPLICATION_JOB_NAME`
- `TEXT_ASSET_REPLICATION_QUEUE_PREFIX`
- `B2_S3_ENDPOINT` (job-level env)
- `B2_BUCKET` (job-level env)
- `B2_REGION` (job-level env)
- `B2_KEY_ID_SECRET` (secret name mapped to `B2_KEY_ID`)
- `B2_APPLICATION_KEY_SECRET` (secret name mapped to `B2_APPLICATION_KEY`)

For async embedding indexing, deploy the embedding queue workflow trigger:

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

Embedding workflow/job env vars:
- `ASSET_EMBEDDING_JOB_NAME`
- `TEXT_ASSET_EMBEDDING_QUEUE_PREFIX`
- `TEXT_ASSET_EMBEDDING_PROCESSED_PREFIX`
- `TEXT_ASSET_EMBEDDING_FAILED_PREFIX`
- `TEXT_ASSET_EMBEDDING_MODEL`
- `VECTOR_STORE_PROVIDER`
- `VECTOR_STORE_PROJECT_ID`
- `VECTOR_STORE_LOCATION`
- `VECTOR_STORE_NAMESPACE`
- `VECTOR_STORE_DIMENSION`
- `VERTEX_INDEX_ENDPOINT`
- `VERTEX_DEPLOYED_INDEX_ID`

Backfill existing catalog assets into the embedding queue:

```bash
BUCKET=<bucket> \
TEXT_ASSET_EMBEDDING_QUEUE_PREFIX=automation/asset_embedding/queue \
TEXT_ASSET_EMBEDDING_BACKFILL_STATE_PREFIX=automation/asset_embedding/backfill \
python tools/asset_catalog/backfill_embeddings.py
```

Recommended secure setup (Secret Manager + job binding):

```bash
bash workflows/setup-backblaze-secrets.sh <project_id>

JOB_SA=$(gcloud run jobs describe asset-replication-job \
  --region=<region> \
  --project=<project_id> \
  --format='value(template.template.serviceAccount)')

gcloud secrets add-iam-policy-binding b2-key-id \
  --project=<project_id> \
  --member="serviceAccount:${JOB_SA}" \
  --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding b2-application-key \
  --project=<project_id> \
  --member="serviceAccount:${JOB_SA}" \
  --role="roles/secretmanager.secretAccessor"

B2_S3_ENDPOINT=https://s3.us-west-000.backblazeb2.com \
B2_BUCKET=<b2-bucket> \
B2_KEY_ID_SECRET=b2-key-id \
B2_APPLICATION_KEY_SECRET=b2-application-key \
bash workflows/setup-asset-replication-trigger.sh <project_id> <bucket> <region>
```

For autonomous 1/day text mode, deploy scheduler + daily workflow:

```bash
cd workflows

TEXT_DAILY_QUOTA=1 \
TEXT_AUTONOMY_TIMEZONE=America/New_York \
TEXT_AUTONOMY_PROVIDER_POLICY=openai_primary \
TEXT_AUTONOMY_TEXT_BACKEND=sage \
TEXT_AUTONOMY_QUALITY_TIER=premium \
bash setup-text-autonomy-scheduler.sh <project_id> <bucket> <region>
```

Daily autonomy env vars:
- `TEXT_AUTONOMY_STATE_PREFIX`
- `TEXT_AUTONOMY_TIMEZONE`
- `TEXT_DAILY_QUOTA`
- `TEXT_DAILY_PAUSE_AFTER_CONSEC_FAILS`
- `TEXT_AUTONOMY_PROVIDER_POLICY`
- `TEXT_AUTONOMY_TEXT_BACKEND`
- `TEXT_AUTONOMY_QUALITY_TIER`
- `TEXT_AUTONOMY_ALLOW_IMAGE_FALLBACK`
- `TEXT_AUTONOMY_SEED_COUNT`
- `TEXT_AUTONOMY_EMITTER_JOB_NAME`
- `TEXT_AUTONOMY_EMITTER_TIMEOUT_SECONDS`
- `TEXT_AUTONOMY_SOURCE_WAIT_TIMEOUT_SECONDS`
- `TEXT_AUTONOMY_SOURCE_WAIT_POLL_SECONDS`

Auto-pause behavior:
- State object: `automation/text_daily/state.json`
- Pause marker: `automation/text_daily/.paused`
- Run summary: `automation/text_daily/runs/<YYYY-MM-DD>/run_summary.json`

## Runtime Readiness Checklist (Stage 1-5)

Required Cloud Run jobs in `us-central1` for VM + Cloud Run hybrid runtime:
- `text-request-emitter-job`
- `text-scene-gen-job`
- `text-scene-adapter-job`
- `asset-replication-job`
- `isaac-lab-job`
- `genie-sim-export-job`
- `arena-export-job`

Not required as Cloud Run jobs in the current Stage 4 architecture:
- `genie-sim-submit-job`
- `genie-sim-gpu-job`

Stage 4 contract:
- `genie-sim-export-pipeline.yaml` requires `genie-sim-export-job` in Cloud Run.
- Submission/execution remains on the Cloud Build + GKE/local runtime path.

Optionally run a canary workflow with a known scene ID:

```bash

gcloud workflows run blueprint-pipeline --data='{ "scene_id": "<scene_id>" }'
```

## Smoke Test (10–15 minutes)

Use this quick-start to validate the pipeline in production mode with a single scene. For a full
end-to-end production validation (no mock fallbacks, quality thresholds, and full artifact checks),
see [docs/PRODUCTION_E2E_VALIDATION.md](PRODUCTION_E2E_VALIDATION.md).

1. Generate mock regen3d fixtures (local staging or seed GCS inputs):

   ```bash
   python fixtures/generate_mock_regen3d.py --output-dir /tmp/regen3d-fixtures
   ```

2. Run a single-scene pipeline in production mode with Genie Sim enabled:

   ```bash
   PIPELINE_ENV=production \
   USE_GENIESIM=true \
   python tools/run_local_pipeline.py \
     --scene-dir /mnt/gcs/scenes/<scene_id> \
     --use-geniesim
   ```

3. Expected artifacts (Genie Sim outputs + episodes + markers):

   - `scenes/<scene_id>/geniesim/scene_graph.json`
   - `scenes/<scene_id>/geniesim/asset_index.json`
   - `scenes/<scene_id>/geniesim/task_config.json`
   - `scenes/<scene_id>/geniesim/job.json`
   - `scenes/<scene_id>/geniesim/merged_scene_manifest.json`
   - `scenes/<scene_id>/geniesim/.geniesim_complete`
   - `scenes/<scene_id>/episodes/.episodes_complete`
   - `scenes/<scene_id>/episodes/geniesim_<job_id>/import_manifest.json`
   - `scenes/<scene_id>/variation_assets/.variation_pipeline_complete`
   - `scenes/<scene_id>/usd/.usd_complete`

4. Firebase/GCS verification:

   ```bash
   python - <<'PY'
from tools.firebase_upload import preflight_firebase_connectivity
print(preflight_firebase_connectivity())
PY

   gsutil ls gs://<bucket>/scenes/<scene_id>/episodes/
   ```

## Canary deployment process

Use the canary pipeline to route a tagged subset of scenes through a new Genie Sim image tag before
full rollout.

1. Tag scenes for canary:
   - Add `tags` (or `scene_tags`) to the per-scene config file at `scenes/<scene_id>/config.json`.
   - Example:

     ```json
     {
       "tags": ["canary", "geniesim-v3.1"]
     }
     ```

2. Deploy the canary workflow definition:

   ```bash
   gcloud workflows deploy canary-pipeline \
     --source=workflows/canary-pipeline.yaml \
     --location=<region>
   ```

3. Trigger the canary workflow with the desired tags + image tag:

   ```bash
   gcloud workflows run canary-pipeline \
     --location=<region> \
     --data='{"data":{"bucket":"<bucket>","name":"scenes/<scene_id>/variation_assets/.variation_pipeline_complete"},"canary_tags":"canary","canary_image_tag":"isaacsim-canary","canary_release_channel":"canary"}'
   ```

4. Monitor the canary run:
   - Watch `scenes/<scene_id>/geniesim/job.json` for `canary` metadata and status.
   - Verify the `geniesim` outputs before expanding tags or raising the canary image to stable.
   - Confirm production Genie Sim jobs enforce `GENIESIM_VALIDATE_FRAMES=1` and `GENIESIM_FAIL_ON_FRAME_VALIDATION=1`.

## Canary rollback steps

If a canary scene fails, the submission job writes a rollback marker at
`scenes/<scene_id>/geniesim/.canary_rollback` containing the failure details and assignment metadata.

1. Confirm the rollback marker exists:

   ```bash
   gsutil cat gs://<bucket>/scenes/<scene_id>/geniesim/.canary_rollback
   ```

2. Disable or narrow canary traffic:
   - Remove the canary tag from `scenes/<scene_id>/config.json`, or
   - Update `canary_tags` / `canary_scene_ids` to exclude the scene.

3. Re-run the standard pipeline (stable image tag) by triggering `genie-sim-export-pipeline`
   or waiting for the next `.variation_pipeline_complete` marker.

## Post-deploy validation

- Confirm recent Cloud Run job executions succeed.
- Validate that Eventarc triggers fire on `.regen3d_complete` markers.
- Spot check GCS outputs for a recent scene.

If anything fails, reference the rollback procedures in [docs/rollback.md](rollback.md).

## Local Genie Sim import poller fallback

When running Genie Sim locally (or when the submit/import steps are decoupled),
you can start a local poller that watches `geniesim/job.json` for
`status=completed` and triggers the import once. The poller writes a
`geniesim/.geniesim_import_triggered` marker so restarts do not repeatedly
trigger imports.

```bash
python tools/run_local_pipeline.py --scene-dir ./scene --import-poller
```

Override the polling interval using the CLI flag or `GENIESIM_IMPORT_POLL_INTERVAL`:

```bash
python tools/run_local_pipeline.py \
  --scene-dir ./scene \
  --import-poller \
  --import-poller-interval 15
```
