# Gemini-Based Segmentation Pipeline

This document describes the new **Gemini-only pipeline** that replaces SAM3 segmentation with Gemini's generative capabilities.

## Overview

The new pipeline eliminates the need for SAM3 segmentation and instead uses **Gemini 3.0 Pro Vision** for both scene analysis and object isolation. This approach offers several advantages:

- **Simplified architecture**: Fewer stages, no complex 3D backprojection
- **Better object understanding**: Gemini can infer complete object geometry from partial views
- **Higher quality outputs**: Generated objects are clean, isolated, and studio-quality
- **No segmentation masks needed**: Direct generative approach skips polygon/mask creation

## Pipeline Stages

### Old Pipeline (SAM3-based)
```
Image Upload
    ↓
[seg-job] SAM3 segmentation + polygon masks
    ↓
[da3-job] Depth estimation
    ↓
[layout-job] 3D scene reconstruction
    ↓
[scale-job] Scale refinement
    ↓
[multiview-job] Crop objects + Gemini refinement
    ↓
[sam3d-job] 3D model generation
```

### New Pipeline (Gemini-based)
```
Image Upload
    ↓
[seg-job] Gemini scene inventory generation
    ↓
[multiview-job] Gemini generative object isolation
    ↓
[sam3d-job] 3D model generation
```

## Stage Details

### 1. Segmentation Job (`seg-job`)

**Purpose**: Analyze the scene image and generate a comprehensive object inventory

**Input**:
- Scene image: `scenes/<sceneId>/images/*.jpg`

**Process**:
- Uses `gemini-3-pro-preview` to analyze the scene
- Identifies all objects, furniture, fixtures, and architectural elements
- Generates structured JSON inventory with:
  - Object IDs, categories, descriptions
  - Approximate locations
  - Spatial relationships

**Output**:
- `scenes/<sceneId>/seg/inventory.json` - Complete scene inventory
- `scenes/<sceneId>/seg/dataset/valid/images/room.jpg` - Copy of source image

**Key Files**:
- `seg-job/run_gemini_inventory.py` - Main inventory generation script
- `seg-job/Dockerfile.gemini` - Lightweight Docker image (no ML models)

**Example Inventory**:
```json
{
  "scene_type": "kitchen",
  "objects": [
    {
      "id": "refrigerator_1",
      "category": "appliance",
      "short_description": "Tall white built-in refrigerator with flat panel doors",
      "approx_location": "front left",
      "relationships": [
        "below cabinet_1",
        "left of counter_1"
      ]
    },
    {
      "id": "cabinet_1",
      "category": "furniture",
      "short_description": "Upper white cabinet with horizontal door",
      "approx_location": "top left",
      "relationships": [
        "above refrigerator_1"
      ]
    }
  ]
}
```

---

### 2. Multiview Job (`multiview-job`)

**Purpose**: Generate isolated, studio-quality renders of each object AND the scene background using Gemini's generative capabilities

**Input**:
- Scene image: `scenes/<sceneId>/seg/dataset/valid/images/room.jpg`
- Inventory: `scenes/<sceneId>/seg/inventory.json`

**Process**:

1. **Generate Layout JSON** (for SAM3D discovery):
   - Creates `scenes/<sceneId>/seg/scene_layout_scaled.json`
   - Includes all objects marked as `must_be_separate_asset: true`
   - Includes special `scene_background` object for static elements

2. **Generate Individual Object Renders**:
   - For each object marked as `must_be_separate_asset: true`:
     1. Build a detailed reconstruction prompt with:
        - Target object details
        - Complete scene context (all objects to exclude)
        - Reconstruction requirements
     2. Call `gemini-3-pro-image-preview` with:
        - Full scene image (not cropped)
        - Reconstruction prompt
        - Request for IMAGE output at 2K resolution
     3. Save the generated isolated object image

3. **Generate Scene Background**:
   - Creates a background image with all separate asset objects removed
   - Preserves static elements: walls, floor, ceiling, built-in furniture
   - Uses inpainting to fill spaces where objects were removed
   - Saves to `scenes/<sceneId>/multiview/obj_scene_background/view_0.png`

**Output**:
- `scenes/<sceneId>/multiview/obj_<id>/view_0.png` - Isolated object renders
- `scenes/<sceneId>/multiview/obj_<id>/generation_meta.json` - Object metadata
- `scenes/<sceneId>/multiview/obj_scene_background/view_0.png` - Scene background render
- `scenes/<sceneId>/multiview/obj_scene_background/generation_meta.json` - Background metadata
- `scenes/<sceneId>/seg/scene_layout_scaled.json` - Layout for SAM3D processing

**Key Files**:
- `multiview-job/run_multiview_gemini_generative.py` - Generative multiview script
- `multiview-job/run_generate_scene_background.py` - Background generation script
- `multiview-job/generate_layout_from_inventory.py` - Layout generation from inventory
- `prompts/object_reconstruction_prompt.md` - Prompt template documentation

**Prompt Strategy**:

For individual objects, the prompt instructs Gemini to:
1. **Isolate** the target object completely (transparent background)
2. **Exclude** all other scene elements and objects
3. **Reconstruct** hidden/occluded parts plausibly
4. **Preserve** materials, colors, and surface details
5. **Render** in orthographic front view with studio lighting
6. **Output** high-resolution PNG with alpha transparency

For scene background, the prompt instructs Gemini to:
1. **Remove** all separate asset objects (appliances, movable items, articulated parts)
2. **Preserve** static elements (walls, floor, ceiling, built-in furniture frames)
3. **Inpaint** removed areas naturally to match surroundings
4. **Reveal** surfaces that were occluded by removed objects
5. **Maintain** original lighting, perspective, and materials
6. **Output** high-resolution RGB image (same view as input)

**Example Output**:
- **Individual Object**: Kitchen scene with refrigerator → Isolated white refrigerator on transparent background
- **Scene Background**: Kitchen scene → Same kitchen view but with refrigerator, kettle, plants removed, showing countertop and walls underneath

---

## Object Classification and Processing

### How Objects Are Classified

During inventory generation (seg-job), Gemini assigns each object two key fields:

1. **`sim_role`**: Simulation role in the scene
   - `scene_shell`: Walls, floor, ceiling, architectural elements
   - `static_furniture_block`: Merged cabinet runs, countertops, built-in furniture
   - `appliance`: Large appliances (refrigerator body, oven body, dishwasher)
   - `articulated_base`: Main body of articulated objects (cabinet frame, appliance body)
   - `articulated_part`: Moving parts (cabinet doors, refrigerator doors, oven door, drawers)
   - `manipulable_object`: Items robots can pick up (dishes, plants, kettles, utensils)
   - `ignore_for_sim`: Decorative clutter not needed for simulation

2. **`must_be_separate_asset`**: Whether to create a separate 3D model
   - `true`: Create individual 3D model (manipulable objects, articulated parts, appliances)
   - `false`: Include in scene background mesh (walls, floors, static furniture blocks)

### Processing Pipeline by Object Type

| Object Type | Example | `must_be_separate_asset` | Processing |
|-------------|---------|--------------------------|------------|
| **Manipulable Objects** | Electric kettle, plant pot, bowl | `true` | Individual 3D model |
| **Articulated Base** | Cabinet frame, refrigerator body | `true` | Individual 3D model |
| **Articulated Part** | Cabinet door, oven door, drawer | `true` | Individual 3D model with `parent_id` |
| **Appliance** | Stovetop (if fixed) | `true` | Individual 3D model |
| **Scene Shell** | Walls, floor, ceiling | `false` | Included in scene background mesh |
| **Static Furniture** | Countertop, cabinet box (if not articulated) | `false` | Included in scene background mesh |
| **Ignore for Sim** | Small decorative items | `false` | Included in scene background mesh |

### Cabinet Example

For a kitchen with upper and lower cabinets:

**If cabinets have visible doors/drawers:**
- Cabinet frame/box: `sim_role: "articulated_base"`, `must_be_separate_asset: true` → Individual 3D model
- Cabinet doors: `sim_role: "articulated_part"`, `must_be_separate_asset: true`, `parent_id: "cabinet_frame"` → Individual 3D models
- Countertop: Often merged with base cabinet or separate depending on simulation needs

**If cabinets are closed/static:**
- Cabinet block: `sim_role: "static_furniture_block"`, `must_be_separate_asset: false` → Included in background mesh
- Countertop: Merged with cabinet block → Included in background mesh

### Complete Scene Reconstruction

The final scene consists of:

1. **Individual 3D Assets** (generated by SAM3D):
   - All objects with `must_be_separate_asset: true`
   - Typically 10-30 objects per scene
   - Can be manipulated, moved, or articulated in simulation

2. **Scene Background Mesh** (generated by SAM3D from background render):
   - Single merged mesh of all static elements
   - Walls, floor, ceiling, built-in furniture
   - Provides collision geometry and visual context
   - Cannot be manipulated in simulation

This approach provides a complete scene reconstruction suitable for robotics simulation, with both:
- **Static environment** for navigation and collision
- **Interactive objects** for manipulation tasks

---

### 3. SAM3D Job (`sam3d-job`)

**Purpose**: Convert isolated object renders AND scene background into 3D models

**Input**:
- Object renders: `scenes/<sceneId>/multiview/obj_*/view_0.png`
- Scene background render: `scenes/<sceneId>/multiview/obj_scene_background/view_0.png`
- Layout: `scenes/<sceneId>/seg/scene_layout_scaled.json`

**Output**:
- Individual object 3D models: `scenes/<sceneId>/assets/obj_*/model.glb`, `model.usdz`
- Scene background 3D model: `scenes/<sceneId>/assets/obj_scene_background/model.glb`, `model.usdz`
- Scene assets manifest: `scenes/<sceneId>/assets/scene_assets.json`

---

## Deployment

### Update Workflow

To use the new Gemini pipeline, deploy the new workflow:

```bash
# Deploy the Gemini-based workflow
gcloud workflows deploy ingest-single-image-pipeline-gemini \
  --source=workflows/ingest-single-image-pipeline-gemini.yaml \
  --location=us-central1
```

### Rebuild Docker Images

The seg-job requires a new lightweight Dockerfile:

```bash
# Build and push new seg-job image
cd seg-job
docker build -f Dockerfile.gemini -t gcr.io/${PROJECT_ID}/seg-job:gemini .
docker push gcr.io/${PROJECT_ID}/seg-job:gemini

# Update seg-job to use new image
gcloud run jobs update seg-job \
  --region=us-central1 \
  --image=gcr.io/${PROJECT_ID}/seg-job:gemini
```

Rebuild multiview-job with the new script:

```bash
# Build and push updated multiview-job
cd multiview-job
docker build -t gcr.io/${PROJECT_ID}/multiview-job:gemini .
docker push gcr.io/${PROJECT_ID}/multiview-job:gemini

# Update multiview-job
gcloud run jobs update multiview-job \
  --region=us-central1 \
  --image=gcr.io/${PROJECT_ID}/multiview-job:gemini
```

### Environment Variables

Ensure these are set:

**seg-job**:
- `GEMINI_API_KEY` - Gemini API key (required)
- `BUCKET` - GCS bucket name
- `IMAGES_PREFIX` - Input images path
- `SEG_PREFIX` - Output segmentation path

**multiview-job**:
- `GEMINI_API_KEY` - Gemini API key (required)
- `BUCKET` - GCS bucket name
- `SCENE_ID` - Scene identifier
- `SEG_PREFIX` - Path to segmentation/inventory
- `MULTIVIEW_PREFIX` - Output multiview path

---

## Migration Guide

### From Old Pipeline to New Pipeline

1. **No code changes needed for SAM3D job** - it consumes the same input format
2. **Remove dependencies on layout/DA3/scale jobs** - these are no longer needed
3. **Update workflow** - use `ingest-single-image-pipeline-gemini.yaml`
4. **Rebuild seg-job and multiview-job** with new images
5. **Test with a sample scene** to validate outputs

### Backward Compatibility

- Old pipeline workflows continue to work unchanged
- New pipeline is deployed as a separate workflow
- Both can coexist during migration period

---

## Advantages Over SAM3 Segmentation

| Aspect | SAM3 Pipeline | Gemini Pipeline |
|--------|---------------|-----------------|
| **Stages** | 6 stages (seg, DA3, layout, scale, multiview, sam3d) | 3 stages (seg, multiview, sam3d) |
| **Segmentation** | Polygon masks + YOLO labels | No masks needed |
| **Object crops** | Polygon-based masking | Generative reconstruction |
| **Hidden geometry** | Not reconstructed | Plausibly inferred |
| **Quality** | Depends on mask quality | Studio-grade isolated renders |
| **Dependencies** | Heavy ML models (SAM3, DepthAnything3) | API calls only |
| **Complexity** | High (3D backprojection, RANSAC, etc.) | Low (inventory + generation) |

---

## Limitations & Considerations

### Limitations

1. **API costs**: Each object requires a Gemini API call with image input
2. **Latency**: Network latency for API calls vs. local inference
3. **Consistency**: Generated objects may vary slightly between runs
4. **Occlusion handling**: Very heavily occluded objects may not reconstruct well

### Best Practices

1. **Scene quality**: Use well-lit, clear scene images
2. **Object visibility**: Works best when objects are at least partially visible
3. **Batch processing**: Consider processing multiple scenes in parallel to amortize latency
4. **Error handling**: Implement retry logic for API failures

---

## Troubleshooting

### Inventory generation fails

- Check `GEMINI_API_KEY` is set correctly
- Verify image is readable and in supported format (JPEG/PNG)
- Check API quotas and rate limits

### Multiview generation produces empty outputs

- Verify inventory.json exists and is valid
- Check that objects have required fields (id, category, description)
- Review Gemini API logs for generation errors

### Generated objects don't match scene

- Review inventory descriptions - ensure they're accurate
- Check if object is heavily occluded in scene
- Consider adjusting prompt temperature (currently 0.3)

---

## Replicator Bundle Generation

After USD assembly completes, the pipeline automatically generates an **NVIDIA Replicator bundle** for synthetic data generation and domain randomization in Isaac Sim.

### What is the Replicator Job?

The `replicator-job` analyzes the completed scene and generates:

1. **Placement Regions** (`placement_regions.usda`): USD layer defining where objects can spawn
2. **Policy-Specific Scripts** (`policies/*.py`): Ready-to-run Replicator Python scripts
3. **Variation Asset Manifests** (`variation_assets/manifest.json`): Additional assets needed
4. **Configuration Files** (`configs/*.json`): Customizable policy parameters

### Supported Environments & Policies

| Environment | Default Policies |
|------------|------------------|
| Kitchen | dish_loading, table_clearing, articulated_access, drawer_manipulation |
| Grocery | grocery_stocking, mixed_sku_logistics, articulated_access |
| Warehouse | mixed_sku_logistics, dexterous_pick_place |
| Lab | precision_insertion, dexterous_pick_place, drawer_manipulation |
| Home Laundry | laundry_sorting, door_manipulation, knob_manipulation |
| Office | drawer_manipulation, panel_interaction |
| Utility Room | panel_interaction, knob_manipulation |

### Output Structure

```
scenes/<scene_id>/replicator/
├── replicator_master.py          # Main entry point
├── placement_regions.usda        # USD layer with placement surfaces
├── bundle_metadata.json          # Scene and policy information
├── policies/                     # Policy-specific scripts
│   ├── dish_loading.py
│   └── ...
├── configs/                      # Policy configurations
│   └── dish_loading.json
└── variation_assets/
    └── manifest.json             # Asset requirements
```

### Using in Isaac Sim

```python
from replicator_master import ReplicatorManager

manager = ReplicatorManager()
manager.list_policies()  # See available policies
manager.run_policy("dish_loading", num_frames=500)
```

### Deployment

```bash
# Build and push replicator-job
cd replicator-job
docker build -t gcr.io/${PROJECT_ID}/replicator-job:latest .
docker push gcr.io/${PROJECT_ID}/replicator-job:latest

# Create Cloud Run job
gcloud run jobs create replicator-job \
  --image=gcr.io/${PROJECT_ID}/replicator-job:latest \
  --region=us-central1 \
  --memory=2Gi \
  --set-env-vars="GEMINI_API_KEY=${GEMINI_API_KEY}"
```

See `replicator-job/README.md` for full documentation.

---

## Future Enhancements

Potential improvements to the Gemini pipeline:

1. **Multi-view generation**: Generate multiple views (front, side, top) per object
2. **Texture enhancement**: Use Gemini to enhance/upscale textures
3. **Batch API calls**: Process multiple objects per API call for efficiency
4. **Prompt optimization**: Fine-tune prompts based on object categories
5. **Quality validation**: Automatically validate generated objects meet criteria
6. **Replicator asset generation**: Auto-generate variation assets using the pipeline

---

## References

- Gemini 3.0 Pro documentation: https://ai.google.dev/gemini-api/docs
- SAM3D repository: https://github.com/facebookresearch/sam-3d-objects
- Workflow syntax: https://cloud.google.com/workflows/docs/reference/syntax
