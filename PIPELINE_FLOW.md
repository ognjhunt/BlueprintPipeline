# Blueprint Pipeline Architecture & Flow

## Overview

This document describes the complete pipeline flow for the Blueprint system, including all triggers, jobs, and data dependencies.

## Pipeline Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 1: Scene Layout Processing                                       │
└─────────────────────────────────────────────────────────────────────────┘

scene_layout_scaled.json uploaded
         │
         │ [Eventarc: multiview-trigger]
         ▼
┌──────────────────────────┐
│  multiview-pipeline      │
│  (Workflow)              │
└──────────────────────────┘
         │
         ├──> 1. multiview-job (Cloud Run Job)
         │    - Generates multiview images for each object
         │    - Output: scenes/<scene_id>/multiview/obj_*/crop.png
         │    - Output: scenes/<scene_id>/multiview/obj_*/*.png
         │
         ├──> 2. Poll for multiview crops (60 retries, 10s interval)
         │    - Waits for crop.png files to appear
         │    - Fails if no crops found within 600 seconds
         │
         └──> 3. assets-plan-job (Cloud Run Job)
              - Reads scene_layout_scaled.json
              - Reads multiview crop images
              - Classifies objects as "interactive" or "static"
              - Creates asset processing plan
              - Output: scenes/<scene_id>/assets/scene_assets.json


┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 2: Parallel Asset Processing (4 pipelines triggered by           │
│          scene_assets.json via Eventarc)                                │
└─────────────────────────────────────────────────────────────────────────┘

scene_assets.json created
         │
         ├────────────────┬────────────────┬────────────────┐
         │                │                │                │
         │ [hunyuan]      │ [sam3d]        │ [interactive]  │ [usd-assembly]
         ▼                ▼                ▼                ▼

┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ hunyuan-     │  │ sam3d-       │  │ interactive- │  │ usd-assembly-│
│ pipeline     │  │ pipeline     │  │ pipeline     │  │ pipeline     │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
       │                 │                 │                 │
       ▼                 ▼                 ▼                 │
┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│ hunyuan-job  │  │ sam3d-job    │  │ interactive- │       ├──> 1. simready-job
│              │  │              │  │ job          │       │    - Prepares assets for sim
│ - Generates  │  │ - 3D recon   │  │              │       │    - Output: prepared assets
│   models     │  │   from multi │  │ - PhysX API  │       │
│   from       │  │   view       │  │   calls      │       ├──> 2. Poll simready-job
│   multiview  │  │ - GLB output │  │ - Interactive│       │    - Waits for completion
│              │  │              │  │   asset data │       │    - Fails if job fails
└──────────────┘  └──────────────┘  └──────────────┘       │
                                                            ▼
                                                     ┌──────────────┐
                                                     │ usd-assembly-│
                                                     │ job          │
                                                     │              │
                                                     │ - GLB→USDZ   │
                                                     │ - Build      │
                                                     │   scene.usda │
                                                     └──────────────┘
```

## Detailed Pipeline Stages

### Stage 1: Multiview Pipeline

**Trigger:** Eventarc trigger `multiview-trigger` watches for:
- Event: `google.cloud.storage.object.v1.finalized`
- Pattern: `scenes/.+/layout/scene_layout_scaled\.json$`

**Workflow:** `multiview-pipeline.yaml`

**Steps:**
1. **Filter**: Only process `scene_layout_scaled.json` files
2. **Derive paths**: Extract scene ID and compute all prefixes
3. **Run multiview-job**:
   - Input: `scene_layout_scaled.json`, segmentation dataset
   - Environment variables:
     - `BUCKET`: GCS bucket name
     - `SCENE_ID`: Scene identifier
     - `LAYOUT_PREFIX`: scenes/<scene_id>/layout
     - `SEG_DATASET_PREFIX`: scenes/<scene_id>/seg/dataset
     - `MULTIVIEW_PREFIX`: scenes/<scene_id>/multiview
     - `ENABLE_GEMINI_VIEWS`: "true"
     - `VIEWS_PER_OBJECT`: "1"
   - Output: Multiview images for each object
4. **Poll for crops**: Wait up to 600 seconds for multiview crops to appear
5. **Run assets-plan-job**:
   - Input: `scene_layout_scaled.json`, multiview crops
   - Environment variables:
     - `BUCKET`: GCS bucket name
     - `SCENE_ID`: Scene identifier
     - `LAYOUT_PREFIX`: scenes/<scene_id>/layout
     - `MULTIVIEW_PREFIX`: scenes/<scene_id>/multiview
     - `ASSETS_PREFIX`: scenes/<scene_id>/assets
   - Output: **`scenes/<scene_id>/assets/scene_assets.json`**

**Critical Output:** `scene_assets.json` triggers all downstream pipelines

### Stage 2: Parallel Asset Processing Pipelines

All four pipelines trigger simultaneously when `scene_assets.json` is created.

#### 2a. Hunyuan Pipeline

**Trigger:** Eventarc trigger `hunyuan-trigger` watches for:
- Event: `google.cloud.storage.object.v1.finalized`
- Pattern: `scenes/.+/assets/scene_assets\.json$`

**Workflow:** `hunyuan-pipeline.yaml`

**Job:** `hunyuan-job`
- **Purpose**: Generate 3D models using Hunyuan AI
- **Input**: `scene_assets.json`, multiview images
- **Process**: AI-based model generation
- **Output**: Generated 3D assets

#### 2b. SAM3D Pipeline

**Trigger:** Eventarc trigger `sam3d-trigger` watches for:
- Event: `google.cloud.storage.object.v1.finalized`
- Pattern: `scenes/.+/assets/scene_assets\.json$`

**Workflow:** `sam3d-pipeline.yaml`

**Job:** `sam3d-job`
- **Purpose**: 3D reconstruction from multiview images
- **Input**: `scene_assets.json`, multiview images
- **Process**: Segment Anything 3D reconstruction
- **Output**: GLB files for static objects

**Note:** Can be commented out/disabled if not needed (per user request)

#### 2c. Interactive Pipeline

**Trigger:** Eventarc trigger `interactive-trigger` watches for:
- Event: `google.cloud.storage.object.v1.finalized`
- Pattern: `scenes/.+/assets/scene_assets\.json$`

**Workflow:** `interactive-pipeline.yaml`

**Job:** `interactive-job`
- **Purpose**: Process interactive objects (doors, drawers, etc.)
- **Input**: `scene_assets.json`, multiview images
- **Process**: Calls PhysX API for articulated objects
- **Output**: Interactive asset data with physics properties
- **Environment**: Requires `PHYSX_ENDPOINT` configuration

#### 2d. USD Assembly Pipeline

**Trigger:** Eventarc trigger `usd-assembly-trigger` watches for:
- Event: `google.cloud.storage.object.v1.finalized`
- Pattern: `scenes/.+/assets/scene_assets\.json$`

**Workflow:** `usd-assembly-pipeline.yaml`

**Jobs (Sequential):**

1. **simready-job**
   - **Purpose**: Prepare assets for simulation
   - **Input**: `scene_assets.json`, processed assets
   - **Process**: Asset preparation and validation
   - **Output**: Simulation-ready assets
   - **Polling**: Workflow polls job status every 10 seconds
   - **Failure handling**: Pipeline fails if job fails

2. **usd-assembly-job** (runs after simready-job completes)
   - **Purpose**: Convert assets to USD format and build scene
   - **Input**: Simready assets, layout data
   - **Process**:
     - Convert GLB files to USDZ
     - Build scene.usda with all assets
     - Apply transforms and metadata
   - **Output**:
     - Individual USDZ files: `scenes/<scene_id>/usd/obj_*/asset.usdz`
     - Scene file: `scenes/<scene_id>/usd/scene.usda`

## Asset Classification Logic

In **assets-plan-job** (`build_scene_assets.py`):

**Interactive Objects:**
- Objects in `INTERACTIVE_OBJECT_IDS` environment variable, OR
- Objects with class names containing:
  - "door", "drawer", "cabinet", "fridge", "oven"
  - "switch", "lever", "hinge", "handle"
  - "button", "knob", "valve"
- Assigned to "physx" pipeline

**Static Objects:**
- All other objects
- Assigned to `STATIC_ASSET_PIPELINE` (default: "sam3d")

## scene_assets.json Format

```json
{
  "scene_id": "scene_123",
  "objects": [
    {
      "id": 0,
      "class_name": "sofa",
      "type": "static",
      "pipeline": "sam3d",
      "multiview_dir": "scenes/scene_123/multiview/obj_0",
      "crop_path": "scenes/scene_123/multiview/obj_0/crop.png",
      "polygon": [[x1, y1], [x2, y2], ...],
      "asset_path": "scenes/scene_123/assets/obj_0/asset.glb"
    },
    {
      "id": 5,
      "class_name": "door",
      "type": "interactive",
      "pipeline": "physx",
      "multiview_dir": "scenes/scene_123/multiview/obj_5",
      "crop_path": "scenes/scene_123/multiview/obj_5/crop.png",
      "polygon": [[x1, y1], [x2, y2], ...],
      "interactive_output": "scenes/scene_123/assets/interactive/obj_5",
      "physx_endpoint": "https://physx-api-endpoint.com"
    }
  ],
  "interactive_count": 3,
  "physx_endpoint": "https://physx-api-endpoint.com"
}
```

## File Path Conventions

```
scenes/
└── <scene_id>/
    ├── layout/
    │   ├── scene_layout.json          # Initial layout (unscaled)
    │   └── scene_layout_scaled.json   # Scaled layout (triggers multiview)
    ├── seg/
    │   └── dataset/
    │       └── data.yaml              # Segmentation dataset
    ├── multiview/
    │   └── obj_<id>/
    │       ├── crop.png               # Main crop (polled by workflow)
    │       └── *.png                  # Additional views
    ├── assets/
    │   ├── scene_assets.json          # Asset plan (triggers 4 pipelines)
    │   ├── obj_<id>/
    │   │   └── asset.glb              # Static object asset
    │   └── interactive/
    │       └── obj_<id>/              # Interactive object data
    └── usd/
        ├── obj_<id>/
        │   └── asset.usdz             # USD format asset
        └── scene.usda                 # Complete scene file
```

## Eventarc Triggers Reference

| Trigger Name | Watches For | Triggers Workflow | Location |
|-------------|-------------|-------------------|----------|
| `multiview-trigger` | `scene_layout_scaled.json` | `multiview-pipeline` | us |
| `hunyuan-trigger` | `scene_assets.json` | `hunyuan-pipeline` | us |
| `sam3d-trigger` | `scene_assets.json` | `sam3d-pipeline` | us |
| `interactive-trigger` | `scene_assets.json` | `interactive-pipeline` | us |
| `usd-assembly-trigger` | `scene_assets.json` | `usd-assembly-pipeline` | us |

## Cloud Run Jobs Reference

| Job Name | Called By | Purpose | Output |
|----------|-----------|---------|--------|
| `multiview-job` | multiview-pipeline | Generate multiview images | crop.png + views |
| `assets-plan-job` | multiview-pipeline | Create asset plan | scene_assets.json |
| `hunyuan-job` | hunyuan-pipeline | AI model generation | 3D models |
| `sam3d-job` | sam3d-pipeline | 3D reconstruction | GLB files |
| `interactive-job` | interactive-pipeline | Interactive assets | Physics data |
| `simready-job` | usd-assembly-pipeline | Prepare for simulation | Ready assets |
| `usd-assembly-job` | usd-assembly-pipeline | Build USD scene | USDZ + scene.usda |

## Debugging Checklist

### Multiview-job succeeded but nothing happens after:

1. ✅ Check if `assets-plan-job` ran:
   ```bash
   gcloud run jobs executions list assets-plan-job --region=us-central1
   ```

2. ✅ Verify `scene_assets.json` was created:
   ```bash
   gsutil ls gs://<bucket>/scenes/<scene_id>/assets/scene_assets.json
   ```

3. ✅ Check if downstream pipelines triggered:
   ```bash
   gcloud workflows executions list hunyuan-pipeline --location=us-central1
   gcloud workflows executions list sam3d-pipeline --location=us-central1
   gcloud workflows executions list interactive-pipeline --location=us-central1
   gcloud workflows executions list usd-assembly-pipeline --location=us-central1
   ```

4. ✅ Verify Eventarc triggers exist:
   ```bash
   gcloud eventarc triggers list --location=us
   ```

### Expected Eventarc triggers:

- ✅ `multiview-trigger` → multiview-pipeline
- ✅ `hunyuan-trigger` → hunyuan-pipeline (⚠️ may be missing)
- ✅ `sam3d-trigger` → sam3d-pipeline (⚠️ may be missing)
- ✅ `interactive-trigger` → interactive-pipeline
- ✅ `usd-assembly-trigger` → usd-assembly-pipeline (⚠️ may be missing)

## Common Issues

### Issue: assets-plan-job doesn't run after multiview-job

**Cause:**
- Job not deployed
- Workflow lacks permission to trigger job
- Job name mismatch

**Solution:**
- Verify job exists: `gcloud run jobs describe assets-plan-job --region=us-central1`
- Check workflow logs for errors
- Ensure service account has `run.jobs.run` permission

### Issue: scene_assets.json created but no downstream pipelines trigger

**Cause:**
- Missing Eventarc triggers
- Trigger pattern mismatch
- Workflow not deployed

**Solution:**
- Deploy missing triggers using Terraform (see `terraform/` directory)
- Verify trigger patterns match file paths
- Check trigger status: `gcloud eventarc triggers describe <trigger-name> --location=us`

### Issue: usd-assembly-job runs before simready-job completes

**Cause:** This shouldn't happen - the workflow enforces sequential execution

**Solution:**
- Check `usd-assembly-pipeline.yaml` polling logic
- Verify workflow execution logs

## Performance Notes

- **Multiview crop polling**: Max 600 seconds (60 retries × 10s)
- **Simready polling**: Indefinite (10s intervals until completion/failure)
- **Parallel execution**: All 4 asset pipelines run simultaneously
- **Sequential execution**: Only simready → usd-assembly within usd-assembly-pipeline

## Configuration Requirements

### Environment Variables (Workflows)

**multiview-pipeline:**
- None (gets project ID from system)

**interactive-pipeline:**
- `PHYSX_ENDPOINT`: PhysX API endpoint URL

### Environment Variables (Jobs)

All jobs receive:
- `BUCKET`: GCS bucket name
- `SCENE_ID`: Scene identifier
- Various `*_PREFIX` variables for input/output paths

**assets-plan-job specific:**
- `STATIC_ASSET_PIPELINE`: "sam3d" (default)
- `INTERACTIVE_OBJECT_IDS`: Comma-separated IDs (optional)
- `PHYSX_ENDPOINT`: Physics endpoint (optional)

## Next Steps

1. **Deploy missing Eventarc triggers:**
   ```bash
   cd terraform
   terraform init
   terraform apply
   ```

2. **Verify all jobs are deployed:**
   ```bash
   gcloud run jobs list --region=us-central1
   ```

3. **Test end-to-end pipeline:**
   - Upload `scene_layout_scaled.json`
   - Monitor all workflow executions
   - Verify final USDZ output

## Additional Resources

- Terraform configuration: `terraform/`
- Workflow definitions: `workflows/`
- Job implementations: `<job-name>/`
