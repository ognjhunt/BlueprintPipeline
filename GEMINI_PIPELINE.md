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

**Purpose**: Generate isolated, studio-quality renders of each object using Gemini's generative capabilities

**Input**:
- Scene image: `scenes/<sceneId>/seg/dataset/valid/images/room.jpg`
- Inventory: `scenes/<sceneId>/seg/inventory.json`

**Process**:
- For each object in the inventory:
  1. Build a detailed reconstruction prompt with:
     - Target object details
     - Complete scene context (all objects to exclude)
     - Reconstruction requirements
  2. Call `gemini-3-pro-image-preview` with:
     - Full scene image (not cropped)
     - Reconstruction prompt
     - Request for IMAGE output at 2K resolution
  3. Save the generated isolated object image

**Output**:
- `scenes/<sceneId>/multiview/obj_<id>/view_0.png` - Isolated object render
- `scenes/<sceneId>/multiview/obj_<id>/generation_meta.json` - Metadata

**Key Files**:
- `multiview-job/run_multiview_gemini_generative.py` - Generative multiview script
- `prompts/object_reconstruction_prompt.md` - Prompt template documentation

**Prompt Strategy**:

The prompt instructs Gemini to:
1. **Isolate** the target object completely (transparent background)
2. **Exclude** all other scene elements and objects
3. **Reconstruct** hidden/occluded parts plausibly
4. **Preserve** materials, colors, and surface details
5. **Render** in orthographic front view with studio lighting
6. **Output** high-resolution PNG with alpha transparency

**Example Output**:
- Input: Kitchen scene with refrigerator among other objects
- Output: Isolated white refrigerator on transparent background, complete with reconstructed sides/back

---

### 3. SAM3D Job (`sam3d-job`)

**Purpose**: Convert isolated object renders into 3D models (unchanged from original pipeline)

**Input**:
- Object renders: `scenes/<sceneId>/multiview/obj_*/view_0.png`

**Output**:
- 3D models: `scenes/<sceneId>/assets/obj_*/model.glb`, `model.usdz`, etc.

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

## Future Enhancements

Potential improvements to the Gemini pipeline:

1. **Multi-view generation**: Generate multiple views (front, side, top) per object
2. **Texture enhancement**: Use Gemini to enhance/upscale textures
3. **Batch API calls**: Process multiple objects per API call for efficiency
4. **Prompt optimization**: Fine-tune prompts based on object categories
5. **Quality validation**: Automatically validate generated objects meet criteria

---

## References

- Gemini 3.0 Pro documentation: https://ai.google.dev/gemini-api/docs
- SAM3D repository: https://github.com/facebookresearch/sam-3d-objects
- Workflow syntax: https://cloud.google.com/workflows/docs/reference/syntax
