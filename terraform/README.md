# Blueprint Pipeline - Eventarc Triggers Infrastructure

This Terraform configuration manages Eventarc triggers for the Blueprint Pipeline.

## Overview

The pipeline follows this flow:

```
multiview-pipeline (triggered by scene_layout_scaled.json)
  ├─> multiview-job
  ├─> assets-plan-job
  └─> creates: scene_assets.json
       │
       ├─> [Eventarc] hunyuan-pipeline → hunyuan-job
       ├─> [Eventarc] sam3d-pipeline → sam3d-job
       ├─> [Eventarc] interactive-pipeline → interactive-job
       └─> [Eventarc] usd-assembly-pipeline
            ├─> simready-job
            └─> usd-assembly-job
```

## What This Terraform Manages

This configuration creates **Eventarc triggers** that watch for `scene_assets.json` file creation and trigger downstream pipelines:

- `hunyuan-trigger` → triggers `hunyuan-pipeline`
- `sam3d-trigger` → triggers `sam3d-pipeline`
- `usd-assembly-trigger` → triggers `usd-assembly-pipeline`
- `interactive-trigger` → triggers `interactive-pipeline`

## Prerequisites

1. **Workflows must be deployed** - The workflows referenced in this config must exist:
   - `hunyuan-pipeline`
   - `sam3d-pipeline`
   - `interactive-pipeline`
   - `usd-assembly-pipeline`

2. **Cloud Run jobs must be deployed** - Each pipeline triggers a Cloud Run job:
   - `hunyuan-job`
   - `sam3d-job`
   - `interactive-job`
   - `simready-job`
   - `usd-assembly-job`

3. **GCS bucket must exist** - The bucket where pipeline data is stored

4. **Terraform installed** - Version >= 1.0

5. **GCP authentication** - `gcloud auth application-default login`

## Setup

1. **Copy the example variables file:**
   ```bash
   cd terraform
   cp terraform.tfvars.example terraform.tfvars
   ```

2. **Edit `terraform.tfvars`** and set your bucket name:
   ```hcl
   bucket_name = "your-actual-bucket-name"
   ```

3. **Initialize Terraform:**
   ```bash
   terraform init
   ```

4. **Review the plan:**
   ```bash
   terraform plan
   ```

5. **Apply the configuration:**
   ```bash
   terraform apply
   ```

## Importing Existing Triggers

If you already have `interactive-trigger` deployed, you can import it:

```bash
terraform import google_eventarc_trigger.interactive_trigger \
  projects/blueprint-8c1ca/locations/us/triggers/interactive-trigger
```

Or, remove the `interactive_trigger` resource from `eventarc_triggers.tf` to avoid managing it with Terraform.

## Verification

After applying, verify the triggers exist:

```bash
gcloud eventarc triggers list --location=us
```

You should see:
- `hunyuan-trigger`
- `sam3d-trigger`
- `usd-assembly-trigger`
- `interactive-trigger`

## Testing the Pipeline

1. **Trigger the pipeline** by uploading a `scene_layout_scaled.json` file:
   ```bash
   gsutil cp scenes/test_scene/layout/scene_layout_scaled.json \
     gs://YOUR_BUCKET/scenes/test_scene/layout/scene_layout_scaled.json
   ```

2. **Monitor the execution:**
   ```bash
   # Check workflow executions
   gcloud workflows executions list multiview-pipeline --location=us-central1

   # Check Cloud Run job executions
   gcloud run jobs executions list multiview-job --region=us-central1
   gcloud run jobs executions list assets-plan-job --region=us-central1
   ```

3. **Verify scene_assets.json was created:**
   ```bash
   gsutil ls gs://YOUR_BUCKET/scenes/test_scene/assets/scene_assets.json
   ```

4. **Check downstream pipelines were triggered:**
   ```bash
   gcloud workflows executions list hunyuan-pipeline --location=us-central1
   gcloud workflows executions list sam3d-pipeline --location=us-central1
   gcloud workflows executions list interactive-pipeline --location=us-central1
   gcloud workflows executions list usd-assembly-pipeline --location=us-central1
   ```

## Troubleshooting

### Triggers not firing

1. Check trigger configuration:
   ```bash
   gcloud eventarc triggers describe hunyuan-trigger --location=us
   ```

2. Verify the bucket name matches in both the trigger and your GCS uploads

3. Check workflow logs:
   ```bash
   gcloud logging read "resource.type=workflows.googleapis.com/Workflow" --limit=50
   ```

### Assets-plan-job not running

The `multiview-pipeline.yaml` workflow should automatically call `assets-plan-job` after `multiview-job` completes. Check:

1. Multiview-pipeline execution logs
2. Assets-plan-job exists and is deployed
3. Workflow has permission to trigger the job

### Scene_assets.json not triggering pipelines

1. Verify file path matches the pattern: `scenes/<scene_id>/assets/scene_assets.json`
2. Check Eventarc trigger logs
3. Ensure workflows have proper permissions

## File Pattern Details

All triggers watch for files matching:
```
^scenes/.+/assets/scene_assets\.json$
```

The workflow filtering logic (in each workflow YAML) ensures only `scene_assets.json` files trigger the pipelines.

## Cost Considerations

- Eventarc triggers: ~$0.40 per million invocations
- Workflows: First 5,000 steps per month are free
- Cloud Run jobs: Billed based on execution time and resources

## Cleanup

To remove all triggers:

```bash
terraform destroy
```

Or manually:
```bash
gcloud eventarc triggers delete hunyuan-trigger --location=us
gcloud eventarc triggers delete sam3d-trigger --location=us
gcloud eventarc triggers delete usd-assembly-trigger --location=us
gcloud eventarc triggers delete interactive-trigger --location=us
```
