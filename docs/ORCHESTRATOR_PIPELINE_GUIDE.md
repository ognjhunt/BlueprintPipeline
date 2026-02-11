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

The orchestrator workflow calls each sub-workflow directly via the `googleapis.workflowexecutions.v1` API. No inter-stage triggers needed. The orchestrator constructs synthetic GCS events matching each sub-workflow's expected input format.

## Pipeline Stages

### Stage 1: VM Reconstruction (~20-45 min)

**What it does:**
1. Checks GPU VM status (`isaac-sim-ubuntu` in `us-east1-c`)
2. Starts the VM if stopped (waits 60s for boot)
3. Launches `run_pipeline_gcs.sh` on the VM via Cloud Build SSH
4. Polls for `.reconstruction_complete` marker in GCS

**What runs on the VM:**
- Downloads the uploaded image from GCS
- Runs 3D-RE-GEN reconstruction pipeline (7 steps):
  1. **Segmentation**: Gemini auto-labeling + GroundingDINO + SAM1 (detects and segments objects)
  2. **Inpainting**: Gemini 2.5 Flash generates empty room + object-isolated images
  3. **Camera estimation**: VGGT monocular depth/camera estimation
  4. **Shape generation**: Hunyuan3D-2.1 generates GLB meshes per object (~34s each)
  5. **Point cloud extraction**: Extracts per-object point clouds from VGGT depth
  6. **Pose matching**: Differentiable rendering aligns GLBs to image (150 iters per object, ~4.5 min each)
  7. **Scene assembly**: Combines all posed GLBs into `combined_scene.glb`
- Adapts reconstruction outputs into BlueprintPipeline `regen3d/` + `assets/` artifacts
- Emits `.regen3d_complete` for Stage 2 (`usd-assembly-pipeline`)
- Uploads results back to GCS

**Known issues:**
- Hunyuan3D-2.1 outputs `{name}_shape.glb` but downstream code expects `{name}.glb`. Patches on VM auto-create symlinks.
- VM service account has `devstorage.read_only` scope — needs `cloud-platform` for GCS writes. Currently uploads via Mac-side gsutil.

**Completion marker:** `scenes/<scene_id>/.reconstruction_complete`

### Stage 2: USD Assembly (~5-15 min)

**Sub-workflow:** `usd-assembly-pipeline`
**Synthetic event:** `{"data": {"bucket": "...", "name": "scenes/<id>/assets/.regen3d_complete"}}`

**What it does:**
- GLB → USDZ conversion
- SimReady asset assembly
- Interactive articulation pass (required for articulated-required scenes)
- Final USD scene assembly with articulation wiring
- Replicator bundle generation (required in strict mode)
- Isaac Lab baseline task package generation

### Stage 3: Variation Assets (~5-15 min)

**Sub-workflow:** `variation-assets-pipeline`
**Synthetic event:** `{"data": {"bucket": "...", "name": "scenes/<id>/replicator/.replicator_complete"}}`

**What it does:**
- Generates scene variations (lighting, materials, object placement)
- Creates SimReady variation assets
- Runs Isaac Lab refresh-only pass to update spawn/task plans from generated variations

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

- Check VM quota in `us-east1-c` (g2-standard-32 needs 32 vCPUs + 1 L4 GPU)
- Check if VM exists: `gcloud compute instances describe isaac-sim-ubuntu --zone=us-east1-c`
- Manual start: `gcloud compute instances start isaac-sim-ubuntu --zone=us-east1-c`
- VM was migrated from `us-east1-b` → `us-east1-d` → `us-east1-c` due to GPU stockouts

### Cloud Build SSH fails (Stage 1)

- Cloud Build SA needs `roles/compute.instanceAdmin.v1` and `roles/compute.osAdminLogin`
- VM must have OS Login enabled
- Check: `gcloud compute ssh isaac-sim-ubuntu --zone=us-east1-c -- "echo ok"`

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
| `workflows/usd-assembly-pipeline.yaml` | Stage 2 sub-workflow (convert → simready → interactive → full USD → replicator → isaac baseline) |
| `workflows/variation-assets-pipeline.yaml` | Stage 3 sub-workflow (variation-gen → simready → isaac refresh-only) |
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
│   ├── .interactive_complete           # Stage 2 interactive completion marker (when run)
│   ├── .interactive_summary.json       # Stage 2 interactive diagnostics
│   ├── *.glb, *.usdz                   # 3D reconstruction outputs
│   └── scene_assets.json               # Asset catalog
├── replicator/
│   ├── .replicator_complete            # Stage 2 completion marker
│   └── ...                             # Replicator renders
├── isaac_lab/
│   ├── .isaac_lab_complete             # Stage 2 baseline Isaac completion marker
│   ├── .isaac_lab_refresh_complete     # Stage 3 refresh completion marker
│   └── ...                             # Isaac task outputs
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

1. **GPU VM** (`isaac-sim-ubuntu`) must exist in `us-east1-c` with 3D-RE-GEN repo at `/home/nijelhunt1/3D-RE-GEN` and venv at `venv_py310`
2. **Cloud Run jobs** for stages 2-4 must be deployed (simready-job, usd-assembly-job, replicator-job, variation-gen-job, genie-sim-export-job, etc.)
3. **Sub-workflows** must be deployed to Cloud Workflows in `us-central1`
4. **GKE cluster** needed for GenieSim GPU jobs (Stage 4)
5. **Service accounts** with correct IAM permissions (handled by `setup-orchestrator-trigger.sh`)

The VM is the only component that requires manual intervention (it auto-starts but may need container warmup). All other stages are fully serverless.

## Local Pipeline Runner (Manual / Dev)

For development and testing, you can run stages 1-3 from your local machine using `run_local_pipeline.py`:

```bash
# Set up environment
export PYTHONPATH="/path/to/BlueprintPipeline"

# Run reconstruction + downstream steps for a prepared scene directory
python3 tools/run_local_pipeline.py \
  --scene-dir /tmp/blueprint_scenes/ChIJHy53k-XlrIkRTdgT1Ev8ln4 \
  --steps regen3d-reconstruct,regen3d,simready,usd

# With GCS upload (requires ADC or service account)
python3 tools/run_local_pipeline.py \
  --scene-dir /tmp/blueprint_scenes/ChIJHy53k-XlrIkRTdgT1Ev8ln4 \
  --steps regen3d-reconstruct \
  --gcs-bucket blueprint-8c1ca.appspot.com \
  --gcs-download-inputs \
  --gcs-upload-outputs
```

### Key Config: `configs/regen3d_reconstruct.env`

```
REGEN3D_VM_HOST=isaac-sim-ubuntu
REGEN3D_VM_ZONE=us-east1-c
REGEN3D_REPO_PATH=/home/nijelhunt1/3D-RE-GEN
REGEN3D_SEG_BACKEND=grounded_sam
REGEN3D_STEPS=1,2,3,4,5,6,7
REGEN3D_REPAIR_EMPTYROOM_PC=false
GCS_BUCKET=blueprint-8c1ca.appspot.com
```

### VM Patches Required

The 3D-RE-GEN code on the VM requires several patches that survive stop/start:

1. **`scene_reconstruction/run.py`** — GLB pre-filter: Hunyuan3D-2.1 outputs `{name}_shape.glb`, not `{name}.glb`. The pre-filter must check both paths and create symlinks.
2. **`pose_matching_planar.py`** — GLB loading fallback for `_shape.glb` suffix.
3. **`global_utils.py`** — Guard against exporting empty scenes (zero geometry RuntimeError).
4. **`pc_utils.py`** — Empty point cloud tensor guard.
5. **`extract_pc_object.py`** — try-except around per-object extraction.

These patches are NOT in git on the VM. If the VM is recreated from an image, re-apply them.

### VM Scopes Issue

The VM currently has `devstorage.read_only` scope. For autonomous GCS uploads, change to `cloud-platform`:

```bash
# VM must be stopped first
gcloud compute instances set-service-account isaac-sim-ubuntu \
  --zone=us-east1-c \
  --scopes=cloud-platform
```
