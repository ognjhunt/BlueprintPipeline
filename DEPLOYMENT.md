# Blueprint Pipeline - Deployment Guide

## The Problem You're Experiencing

**Symptom:** multiview-job succeeds but nothing happens after.

**Root Cause:** **assets-plan-job is not deployed!**

Evidence:
- `ingest-single-image-pipeline.yaml:303` says: *"We skip assets-plan-job as it is not in your list of successfully deployed jobs"*
- `multiview-pipeline.yaml:107` tries to call assets-plan-job, but it doesn't exist
- When the job call fails, `scene_assets.json` is never created
- Without `scene_assets.json`, downstream pipelines (hunyuan, sam3d, interactive, usd-assembly) never trigger

## Quick Fix

```bash
# Option 1: Deploy just assets-plan-job
chmod +x deploy-assets-plan-job.sh
./deploy-assets-plan-job.sh

# Option 2: Deploy all jobs
chmod +x deploy-all-jobs.sh
./deploy-all-jobs.sh
```

Then deploy the Eventarc triggers:
```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars and set your bucket_name
terraform init
terraform apply
```

## Complete Deployment Steps

### Prerequisites

1. **Docker** installed and running
2. **gcloud CLI** installed and authenticated:
   ```bash
   gcloud auth login
   gcloud auth configure-docker
   gcloud config set project blueprint-8c1ca
   ```
3. **Terraform** installed (>= 1.0)
4. **Permissions** required:
   - Cloud Run Admin
   - Storage Admin
   - Workflows Admin
   - Eventarc Admin
   - Container Registry Admin

### Step 1: Deploy Cloud Run Jobs

**Critical Jobs** (must be deployed for pipeline to work):

| Job | Purpose | Required By |
|-----|---------|-------------|
| **assets-plan-job** | Creates scene_assets.json | multiview-pipeline ⚠️ **MISSING** |
| multiview-job | Generates multiview images | multiview-pipeline |
| hunyuan-job | AI model generation | hunyuan-pipeline |
| interactive-job | Interactive assets | interactive-pipeline |
| simready-job | Prepare for simulation | usd-assembly-pipeline |
| usd-assembly-job | Build USD scene | usd-assembly-pipeline |

**Deploy all jobs:**

```bash
# Make script executable
chmod +x deploy-all-jobs.sh

# Set environment (optional, defaults are shown)
export PROJECT_ID=blueprint-8c1ca
export REGION=us-central1

# Deploy all jobs
./deploy-all-jobs.sh
```

This will deploy jobs in the correct order with appropriate resource allocation.

**Deploy only assets-plan-job:**

```bash
chmod +x deploy-assets-plan-job.sh
./deploy-assets-plan-job.sh
```

**Manual deployment (if scripts fail):**

```bash
cd assets-plan

# Build and push image
docker build -t gcr.io/blueprint-8c1ca/assets-plan-job:latest .
docker push gcr.io/blueprint-8c1ca/assets-plan-job:latest

# Create Cloud Run job
gcloud run jobs create assets-plan-job \
  --image=gcr.io/blueprint-8c1ca/assets-plan-job:latest \
  --region=us-central1 \
  --max-retries=0 \
  --task-timeout=1800 \
  --memory=2Gi \
  --cpu=2

cd ..
```

**Verify jobs are deployed:**

```bash
gcloud run jobs list --region=us-central1
```

You should see:
- ✅ assets-plan-job
- ✅ multiview-job
- ✅ hunyuan-job
- ✅ sam3d-job
- ✅ interactive-job
- ✅ simready-job
- ✅ usd-assembly-job

### Step 2: Deploy Workflows

Workflows define the pipeline orchestration.

```bash
cd workflows

# Deploy each workflow
for workflow in *.yaml; do
  name=$(basename "$workflow" .yaml)
  echo "Deploying $name..."

  gcloud workflows deploy "$name" \
    --source="$workflow" \
    --location=us-central1 \
    --service-account=744608654760-compute@developer.gserviceaccount.com
done

cd ..
```

**Verify workflows are deployed:**

```bash
gcloud workflows list --location=us-central1
```

You should see:
- ✅ multiview-pipeline
- ✅ hunyuan-pipeline
- ✅ sam3d-pipeline
- ✅ interactive-pipeline
- ✅ usd-assembly-pipeline
- Plus other pipelines

### Step 3: Deploy Eventarc Triggers

Triggers connect Cloud Storage events to workflows.

```bash
cd terraform

# Copy example variables
cp terraform.tfvars.example terraform.tfvars

# Edit terraform.tfvars
nano terraform.tfvars
# Set: bucket_name = "your-actual-bucket-name"

# Initialize Terraform
terraform init

# Review what will be created
terraform plan

# Deploy triggers
terraform apply

cd ..
```

**Verify triggers are deployed:**

```bash
gcloud eventarc triggers list --location=us
```

You should see:
- ✅ multiview-trigger (should already exist)
- ✅ hunyuan-trigger
- ✅ sam3d-trigger
- ✅ interactive-trigger
- ✅ usd-assembly-trigger

### Step 4: Test the Pipeline

**Upload a test scene:**

```bash
# Replace with your actual bucket and scene
BUCKET="your-bucket-name"
SCENE_ID="test_scene"

# Upload scene_layout_scaled.json to trigger the pipeline
gsutil cp "scenes/${SCENE_ID}/layout/scene_layout_scaled.json" \
  "gs://${BUCKET}/scenes/${SCENE_ID}/layout/scene_layout_scaled.json"
```

**Monitor execution:**

```bash
# Watch multiview-pipeline
gcloud workflows executions list multiview-pipeline --location=us-central1 --limit=1

# Get the execution ID from above, then:
EXEC_ID="<execution-id>"
gcloud workflows executions describe "$EXEC_ID" \
  --workflow=multiview-pipeline \
  --location=us-central1

# Check if multiview-job ran
gcloud run jobs executions list multiview-job --region=us-central1 --limit=1

# CRITICAL: Check if assets-plan-job ran (this was failing before!)
gcloud run jobs executions list assets-plan-job --region=us-central1 --limit=1

# Check if scene_assets.json was created
gsutil ls "gs://${BUCKET}/scenes/${SCENE_ID}/assets/scene_assets.json"

# If scene_assets.json exists, check downstream pipelines
gcloud workflows executions list hunyuan-pipeline --location=us-central1 --limit=1
gcloud workflows executions list sam3d-pipeline --location=us-central1 --limit=1
gcloud workflows executions list interactive-pipeline --location=us-central1 --limit=1
gcloud workflows executions list usd-assembly-pipeline --location=us-central1 --limit=1
```

**Expected flow:**

```
1. Upload scene_layout_scaled.json
   ↓
2. [Eventarc] multiview-trigger fires
   ↓
3. multiview-pipeline executes
   ├─> multiview-job runs (creates crops)
   ├─> Poll for crops (wait up to 600s)
   └─> assets-plan-job runs ⚠️ (previously failed - job didn't exist)
       ↓
       Creates: scene_assets.json
       ↓
4. [Eventarc] Four triggers fire simultaneously:
   ├─> hunyuan-trigger → hunyuan-pipeline → hunyuan-job
   ├─> sam3d-trigger → sam3d-pipeline → sam3d-job
   ├─> interactive-trigger → interactive-pipeline → interactive-job
   └─> usd-assembly-trigger → usd-assembly-pipeline
       ├─> simready-job
       └─> usd-assembly-job
```

## Troubleshooting

### assets-plan-job execution fails

Check logs:
```bash
EXEC_ID=$(gcloud run jobs executions list assets-plan-job --region=us-central1 --limit=1 --format="value(name)")
gcloud run jobs executions describe "$EXEC_ID" --region=us-central1
gcloud logging read "resource.type=cloud_run_job AND resource.labels.job_name=assets-plan-job" --limit=50
```

Common issues:
- **Missing environment variables**: Check workflow passes all required env vars
- **Bucket permissions**: Ensure service account has Storage Object Admin
- **Input files missing**: Verify `scene_layout_scaled.json` and multiview crops exist

### multiview-pipeline fails at polling step

```bash
# Check if multiview crops were created
gsutil ls "gs://${BUCKET}/scenes/${SCENE_ID}/multiview/"
gsutil ls "gs://${BUCKET}/scenes/${SCENE_ID}/multiview/obj_*/crop.png"
```

If no crops exist:
- multiview-job failed or didn't complete
- Check multiview-job logs

### Downstream pipelines don't trigger

1. **Verify scene_assets.json exists:**
   ```bash
   gsutil ls "gs://${BUCKET}/scenes/${SCENE_ID}/assets/scene_assets.json"
   ```

2. **Check Eventarc triggers:**
   ```bash
   gcloud eventarc triggers list --location=us
   ```

3. **Check trigger logs:**
   ```bash
   gcloud logging read "resource.type=eventarc.googleapis.com/Trigger" --limit=50
   ```

4. **Manually trigger a workflow (for testing):**
   ```bash
   gcloud workflows run hunyuan-pipeline \
     --location=us-central1 \
     --data='{"data":{"bucket":"your-bucket","name":"scenes/test/assets/scene_assets.json"}}'
   ```

## Resource Requirements

| Job | Memory | CPU | Timeout | Notes |
|-----|--------|-----|---------|-------|
| assets-plan-job | 2Gi | 2 | 1800s | Lightweight planner |
| multiview-job | 4Gi | 4 | 3600s | AI model inference |
| hunyuan-job | 8Gi | 4 | 7200s | Heavy AI workload |
| sam3d-job | 8Gi | 4 | 7200s | 3D reconstruction |
| interactive-job | 4Gi | 2 | 3600s | API calls |
| simready-job | 2Gi | 2 | 1800s | Asset preparation |
| usd-assembly-job | 4Gi | 2 | 3600s | USD conversion |

Adjust in `deploy-all-jobs.sh` if needed.

## Cost Estimate

**Per scene execution:**

- Cloud Run jobs: ~$0.10-$0.50 (depends on scene complexity)
- Workflows: Free (under 5000 steps/month)
- Eventarc: ~$0.00 (minimal invocations)
- Storage: ~$0.02/GB/month

**Total estimated cost:** ~$0.20-$0.75 per scene

## Security Notes

- Service account: `744608654760-compute@developer.gserviceaccount.com`
- Required IAM roles:
  - `roles/run.invoker` - Trigger Cloud Run jobs
  - `roles/storage.objectAdmin` - Read/write GCS
  - `roles/logging.logWriter` - Write logs

## Next Steps After Deployment

1. ✅ All Cloud Run jobs deployed
2. ✅ All workflows deployed
3. ✅ All Eventarc triggers created
4. ✅ Test pipeline end-to-end
5. 📝 Monitor costs and optimize resource allocation
6. 📝 Set up alerting for failed executions
7. 📝 Document scene data requirements

## Quick Reference Commands

```bash
# List all jobs
gcloud run jobs list --region=us-central1

# List all workflows
gcloud workflows list --location=us-central1

# List all triggers
gcloud eventarc triggers list --location=us

# Tail logs for a specific job
gcloud logging tail "resource.type=cloud_run_job AND resource.labels.job_name=assets-plan-job"

# Delete a job (careful!)
gcloud run jobs delete <job-name> --region=us-central1

# Delete a trigger (careful!)
gcloud eventarc triggers delete <trigger-name> --location=us
```

## Support

If issues persist:
1. Check workflow execution logs
2. Check Cloud Run job execution logs
3. Verify all files exist in GCS at expected paths
4. Ensure service account has correct permissions
5. Review `PIPELINE_FLOW.md` for architecture details
