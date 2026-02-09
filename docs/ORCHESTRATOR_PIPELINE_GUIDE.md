# Image-to-Scene Orchestrator Pipeline Guide

End-to-end reference for the single-trigger orchestrator that converts an uploaded image into a fully assembled scene with GenieSim data.

## Architecture Overview

```
                    Image Upload to GCS
                          │
                    ┌─────▼──────┐
                    │  Eventarc   │  (1 trigger: image-upload-orchestrator-trigger)
                    │  Trigger    │  location: us
                    └─────┬──────┘
                          │
            ┌─────────────▼─────────────┐
            │  image-to-scene-orchestrator │  (Cloud Workflows, us-central1)
            │                             │
            │  Stage 1: VM Reconstruction │  Cloud Build → SSH → run_pipeline_gcs.sh
            │  Stage 2: USD Assembly      │  → usd-assembly-pipeline (sub-workflow)
            │  Stage 3: Variation Assets  │  → variation-assets-pipeline (sub-workflow)
            │  Stage 4: GenieSim Export   │  → genie-sim-export-pipeline (sub-workflow)
            │  Stage 5: Arena Export      │  → arena-export-pipeline (non-blocking)
            └─────────────┬─────────────┘
                          │
                    Scene Complete
```

**One trigger, one workflow, five stages.** Upload a `.png`/`.jpg`/`.jpeg` image to `gs://<bucket>/scenes/<scene_id>/images/` and everything runs automatically through to GenieSim output.

## How It Works

### Before: Multi-Trigger Chain (Replaced)

Previously, each pipeline stage wrote a GCS marker file (e.g., `.regen3d_complete`) that fired an Eventarc trigger to start the next stage. This consumed 10+ of the 10 available GCS Pub/Sub notification slots per bucket and broke whenever a trigger was missing.

### Now: Single Orchestrator

The orchestrator workflow calls each sub-workflow directly via the `googleapis.workflowexecutions.v1` API. No inter-stage triggers needed. Each sub-workflow YAML is **unchanged** — the orchestrator constructs synthetic GCS events matching each sub-workflow's expected input format.

## Pipeline Stages

### Stage 1: VM Reconstruction (~20-45 min)

**What it does:**
1. Checks GPU VM status (`isaac-sim-ubuntu` in `us-east1-b`)
2. Starts the VM if stopped (waits 60s for boot)
3. Launches `run_pipeline_gcs.sh` on the VM via Cloud Build SSH
4. Polls for `.reconstruction_complete` marker in GCS

**What runs on the VM:**
- Downloads the uploaded image from GCS
- Runs 3D-RE-GEN reconstruction (NeRF → mesh)
- Generates SimReady assets (materials, physics)
- Assembles initial USD scene
- Uploads results back to GCS

**Completion marker:** `scenes/<scene_id>/.reconstruction_complete`

### Stage 2: USD Assembly (~5-15 min)

**Sub-workflow:** `usd-assembly-pipeline`
**Synthetic event:** `{"data": {"bucket": "...", "name": "scenes/<id>/assets/.regen3d_complete"}}`

**What it does:**
- GLB → USDZ conversion
- SimReady asset assembly
- Replicator rendering
- Isaac Lab scene setup

### Stage 3: Variation Assets (~5-15 min)

**Sub-workflow:** `variation-assets-pipeline`
**Synthetic event:** `{"data": {"bucket": "...", "name": "scenes/<id>/replicator/.replicator_complete"}}`

**What it does:**
- Generates scene variations (lighting, materials, object placement)
- Creates SimReady variation assets

### Stage 4: GenieSim Export (~10-30 min)

**Sub-workflow:** `genie-sim-export-pipeline`
**Synthetic event:** `{"data": {"bucket": "...", "name": "scenes/<id>/variation_assets/.variation_pipeline_complete"}}`

**What it does:**
- Exports scene data for GenieSim consumption
- Launches GPU job on GKE cluster
- Generates robot manipulation episodes

### Stage 5: Arena Export (non-blocking, ~5-10 min)

**Sub-workflow:** `arena-export-pipeline`
**Synthetic event:** `{"data": {"bucket": "...", "name": "scenes/<id>/geniesim/.geniesim_complete"}}`

**What it does:**
- Exports final scene data to Arena format
- Failures here do NOT abort the pipeline (best-effort)

## Deployment Details

### Active Resources

| Resource | Name | Location |
|----------|------|----------|
| Workflow | `image-to-scene-orchestrator` | `us-central1` |
| Eventarc Trigger | `image-upload-orchestrator-trigger` | `us` |
| Service Account | `workflow-invoker@<project>.iam.gserviceaccount.com` | — |
| Bucket | `blueprint-8c1ca.appspot.com` | Multi-region US |

### Why Trigger Is in `us` and Workflow in `us-central1`

The GCS bucket is multi-region `US`. Eventarc triggers must be in the same location as the bucket, so the trigger lives in `us`. The workflow itself is deployed to `us-central1` (a specific region within US). The trigger's `--destination-workflow-location=us-central1` bridges the two.

### IAM Permissions

The `workflow-invoker` service account has:
- `roles/workflows.invoker` — execute workflows
- `roles/run.invoker` — invoke Cloud Run jobs
- `roles/storage.objectAdmin` — read/write GCS markers
- `roles/logging.logWriter` — structured logging
- `roles/compute.instanceAdmin.v1` — start/stop VM
- `roles/cloudbuild.builds.editor` — launch Cloud Build
- `roles/eventarc.eventReceiver` — receive Eventarc events

The Cloud Build service account also needs:
- `roles/compute.instanceAdmin.v1` — SSH to VM
- `roles/compute.osAdminLogin` — OS login for SSH

The GCS service agent needs:
- `roles/pubsub.publisher` — publish events to Pub/Sub for Eventarc

## Setup / Redeploy

### First-time setup or full redeploy

```bash
cd workflows/

# 1. Deploy orchestrator and create trigger
bash setup-orchestrator-trigger.sh [project_id] [bucket_name] [workflow_region]
# Defaults: project from gcloud config, bucket = <project>.appspot.com, region = us-central1

# 2. (Optional) Clean up old multi-trigger chain
bash cleanup-old-triggers.sh [project_id] [trigger_location]
# Defaults: project from gcloud config, trigger_location = us
```

### Redeploy workflow only (after editing YAML)

```bash
gcloud workflows deploy image-to-scene-orchestrator \
  --location=us-central1 \
  --source=workflows/image-to-scene-orchestrator.yaml \
  --service-account=workflow-invoker@<project>.iam.gserviceaccount.com \
  --project=<project>
```

### Verify current state

```bash
# Check workflow is active
gcloud workflows describe image-to-scene-orchestrator --location=us-central1

# Check trigger exists
gcloud eventarc triggers describe image-upload-orchestrator-trigger --location=us

# Check recent executions
gcloud workflows executions list image-to-scene-orchestrator --location=us-central1 --limit=5

# Check GCS notification slots (max 10)
gsutil notification list gs://blueprint-8c1ca.appspot.com | grep -c "notificationConfigs"
```

## Testing

### Upload an image (triggers full pipeline)

```bash
gsutil cp photo.jpeg gs://blueprint-8c1ca.appspot.com/scenes/test_scene/images/kitchen.jpeg
```

### Manual workflow execution (without trigger)

```bash
gcloud workflows run image-to-scene-orchestrator --location=us-central1 \
  --data='{"data":{"bucket":"blueprint-8c1ca.appspot.com","name":"scenes/test_scene/images/kitchen.jpeg","generation":"1234567890"}}'
```

### Monitor execution

```bash
# List recent executions
gcloud workflows executions list image-to-scene-orchestrator \
  --location=us-central1 --limit=5

# Watch a specific execution
gcloud workflows executions describe <execution_id> \
  --workflow=image-to-scene-orchestrator --location=us-central1

# Check Cloud Build logs (Stage 1)
gcloud builds list --project=<project> --limit=5
gcloud builds log <build_id>

# Check structured logs
gcloud logging read 'resource.type="workflows.googleapis.com/Workflow" AND textPayload=~"orchestrator"' \
  --project=<project> --limit=20 --format="table(timestamp,textPayload)"
```

## Failure Handling

### Stages 1-4: Abort on failure

If any of the first four stages fail, the orchestrator:
1. Writes `.orchestrator_failed` marker to `gs://<bucket>/scenes/<scene_id>/`
2. Emits structured failure metrics (logged as `bp_metric: job_invocation`)
3. Raises an error, marking the workflow execution as FAILED

The failure marker is a JSON file containing:
```json
{
  "scene_id": "test_scene",
  "status": "failed",
  "timestamp": "2026-02-09T...",
  "input_object": "scenes/test_scene/images/kitchen.jpeg",
  "input_generation": "1234567890",
  "error": {
    "code": "subworkflow_failed",
    "message": "usd-assembly-pipeline failed for scene test_scene (state: FAILED)",
    "type": "orchestrator_failure",
    "stage": "usd_assembly"
  },
  "context": {
    "workflow_execution_id": "...",
    "build_id": "...",
    "elapsed_seconds": 120,
    "completed_stages": "reconstruction"
  }
}
```

### Stage 5: Non-blocking failure

Arena export failures are logged as warnings but do NOT fail the overall pipeline. The orchestrator returns SUCCESS with `arena_export: "FAILED"` in the output.

### Deduplication

The orchestrator checks for an existing `.reconstruction_complete` marker. If the marker's `input_generation` matches or exceeds the incoming event's generation, the event is skipped (logged as "stale/duplicate"). This prevents re-running reconstruction when an image is overwritten or re-uploaded.

## Troubleshooting

### Workflow execution failed

```bash
# Get execution details with error
gcloud workflows executions describe <execution_id> \
  --workflow=image-to-scene-orchestrator --location=us-central1 \
  --format="yaml(state,error,result)"

# Check which stage failed (in the error message)
# Format: "[stage_name] error description"
```

### VM won't start (Stage 1)

- Check VM quota in `us-east1-b` (g2-standard-32 needs 32 vCPUs + 1 L4 GPU)
- Check if VM exists: `gcloud compute instances describe isaac-sim-ubuntu --zone=us-east1-b`
- Manual start: `gcloud compute instances start isaac-sim-ubuntu --zone=us-east1-b`

### Cloud Build SSH fails (Stage 1)

- Cloud Build SA needs `roles/compute.instanceAdmin.v1` and `roles/compute.osAdminLogin`
- VM must have OS Login enabled
- Check: `gcloud compute ssh isaac-sim-ubuntu --zone=us-east1-b -- "echo ok"`

### Sub-workflow fails (Stages 2-4)

Check the sub-workflow's own execution:
```bash
# List executions of the failed sub-workflow
gcloud workflows executions list usd-assembly-pipeline --location=us-central1 --limit=5

# The orchestrator error message includes the sub-workflow state
```

### Nothing triggers on image upload

1. Verify trigger exists: `gcloud eventarc triggers describe image-upload-orchestrator-trigger --location=us`
2. Verify bucket notifications: `gsutil notification list gs://blueprint-8c1ca.appspot.com`
3. Check the image path matches: `scenes/<scene_id>/images/<name>.{png|jpg|jpeg}`
4. Check trigger logs: `gcloud logging read 'resource.type="audited_resource" AND protoPayload.serviceName="eventarc.googleapis.com"' --limit=10`

### GCS notification quota (10 per bucket)

Each Eventarc trigger on a GCS bucket consumes one notification slot. If you hit the limit:
```bash
# See current notifications
gsutil notification list gs://blueprint-8c1ca.appspot.com

# Delete a specific notification
gsutil notification delete projects/_/buckets/<bucket>/notificationConfigs/<id>
```

## File Reference

| File | Purpose |
|------|---------|
| `workflows/image-to-scene-orchestrator.yaml` | Master orchestrator workflow (915 lines) |
| `workflows/setup-orchestrator-trigger.sh` | Deploy workflow + create Eventarc trigger |
| `workflows/cleanup-old-triggers.sh` | Delete replaced multi-trigger chain triggers |
| `workflows/usd-assembly-pipeline.yaml` | Stage 2 sub-workflow (unchanged) |
| `workflows/variation-assets-pipeline.yaml` | Stage 3 sub-workflow (unchanged) |
| `workflows/genie-sim-export-pipeline.yaml` | Stage 4 sub-workflow (unchanged) |
| `workflows/arena-export-pipeline.yaml` | Stage 5 sub-workflow (unchanged) |
| `workflows/image-to-scene-pipeline.yaml` | Legacy Stage 1 standalone (kept for independent use) |

## Cloud Workflows Gotchas

These are pitfalls encountered during development. Reference for anyone editing the YAML.

### 1. Inline map literals in expressions

**WRONG** (parse error):
```yaml
- call: some.api
  args:
    body: '${json.encode({"key": value, "key2": value2})}'
```

**RIGHT** (build map as variable first):
```yaml
- build_arg:
    assign:
      - myData:
          key: ${value}
          key2: ${value2}
- call: some.api
  args:
    body: '${json.encode(myData)}'
```

### 2. Variable scoping in except blocks

Variables declared inside `except: as: e` are only accessible within that except block's steps. You cannot reference `e.message` in steps outside the except.

### 3. Trigger location must match bucket location

Multi-region `US` bucket → trigger in `us` (not `us-central1`). The workflow can be in `us-central1` — use `--destination-workflow-location` to bridge.

### 4. Path pattern filters not supported for GCS events

`--event-filters-path-pattern` doesn't work with `google.cloud.storage.object.v1.finalized`. Filter inside the workflow instead (regex on `event.data.name`).

## GCS Data Layout

After a successful pipeline run for scene `my_scene`:

```
gs://bucket/scenes/my_scene/
├── images/
│   └── kitchen.jpeg                    # Uploaded image (trigger input)
├── assets/
│   ├── .regen3d_complete               # Stage 1 completion marker
│   ├── *.glb, *.usdz                   # 3D reconstruction outputs
│   └── scene_assets.json               # Asset catalog
├── replicator/
│   ├── .replicator_complete            # Stage 2 completion marker
│   └── ...                             # Replicator renders
├── variation_assets/
│   ├── .variation_pipeline_complete    # Stage 3 completion marker
│   └── ...                             # Scene variations
├── geniesim/
│   ├── .geniesim_complete              # Stage 4 completion marker
│   └── ...                             # GenieSim export data
├── .reconstruction_complete            # Overall reconstruction marker
└── .orchestrator_failed                # Written on failure (deleted on retry)
```

## Prerequisites

For the full pipeline to run end-to-end:

1. **GPU VM** (`isaac-sim-ubuntu`) must exist in `us-east1-b` with Docker and `run_pipeline_gcs.sh`
2. **Cloud Run jobs** for stages 2-4 must be deployed (simready-job, usd-assembly-job, replicator-job, variation-gen-job, genie-sim-export-job, etc.)
3. **Sub-workflows** must be deployed to Cloud Workflows in `us-central1`
4. **GKE cluster** needed for GenieSim GPU jobs (Stage 4)
5. **Service accounts** with correct IAM permissions (handled by `setup-orchestrator-trigger.sh`)

The VM is the only component that requires manual intervention (it auto-starts but may need container warmup). All other stages are fully serverless.
