# Hunyuan3D Job

This job generates 3D assets (mesh + texture) from 2D images using Tencent's
[Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) model.

## Overview

The Hunyuan job takes asset plans (from the assets-plan-job) and generates:
- **Shape generation**: `mesh.glb` - untextured 3D geometry
- **Texture generation**: `model.glb` / `asset.glb` - textured 3D models
- **Optional USDZ export**: `model.usdz` - for Apple platforms

## Model Assets

* Inference code is cloned from the Hunyuan3D-2.1 repository at `/app/Hunyuan3D-2.1`
* Models are downloaded from Hugging Face (controlled via `HUNYUAN_MODEL_PATH`, default: `tencent/Hunyuan3D-2.1`)
* Models are cached in GCS at `/mnt/gcs/hf-cache` to speed up subsequent runs

Authentication for Hugging Face is provided via `HUGGINGFACE_TOKEN` or `HF_TOKEN`.

## Configuration

### Model Settings:
- `HUNYUAN_MODEL_PATH`: HuggingFace model path (default: `tencent/Hunyuan3D-2.1`)
- `HF_TOKEN`: HuggingFace token for authenticated downloads

### Shape Generation:
- `HUNYUAN_NUM_STEPS`: Number of inference steps (default: 50)
- `HUNYUAN_MAX_NUM_VIEW`: Number of views for multi-view generation (default: 6)
- `HUNYUAN_RESOLUTION`: Input resolution (default: 512)

### Texture Generation:
- `HUNYUAN_RENDER_SIZE`: Render resolution (default: 1024, options: 512/1024/2048)
- `HUNYUAN_TEXTURE_SIZE`: Texture map size (default: 2048, options: 1024/2048/4096)

### Performance:
- `ENABLE_USDZ_EXPORT`: Enable USDZ conversion (default: 1, set to 0 to disable)
- `SKIP_EXISTING_ASSETS`: Skip objects with existing asset.glb (default: 1)

## Performance Optimization

⚠️ **IMPORTANT**: The job may timeout with default settings when processing many objects.

See **[PERFORMANCE.md](./PERFORMANCE.md)** for:
- Detailed performance analysis and bottlenecks
- Quick fixes to reduce processing time
- Environment variable tuning guide
- Model pre-caching instructions
- Batch processing strategies

**Quick Fix** (increase timeout to 2 hours):
```bash
./update-job-timeout.sh
```

## Deployment

Build and push the Docker image:
```bash
cd hunyuan-job
gcloud builds submit --tag gcr.io/blueprint-8c1ca/hunyuan-job:latest
```

Update the Cloud Run job:
```bash
gcloud run jobs update hunyuan-job \
  --region=us-central1 \
  --image=gcr.io/blueprint-8c1ca/hunyuan-job:latest \
  --timeout=7200 \
  --set-env-vars="HUNYUAN_RENDER_SIZE=512,HUNYUAN_TEXTURE_SIZE=1024"
```

## Troubleshooting

### Job times out after 1 hour
- Run `./update-job-timeout.sh` to increase timeout to 2 hours
- Reduce quality settings: `HUNYUAN_RENDER_SIZE=512`, `HUNYUAN_TEXTURE_SIZE=1024`
- See [PERFORMANCE.md](./PERFORMANCE.md) for more optimization tips

### USDZ export fails
- The job now uses USD Python API (`pxr` module) instead of `usd_from_gltf` CLI
- If USDZ is not needed, disable it: `ENABLE_USDZ_EXPORT=0`
- USDZ is only needed for Apple platforms (ARKit, Vision Pro, etc.)

### "Model not found" or download errors
- Ensure `HF_TOKEN` is set in the Cloud Run job environment
- Check that the model path is correct: `HUNYUAN_MODEL_PATH=tencent/Hunyuan3D-2.1`
- Verify HuggingFace token has access to the model repository

### Out of memory errors
- Reduce quality settings to lower VRAM usage
- Use smaller texture sizes: `HUNYUAN_TEXTURE_SIZE=1024`
- Reduce number of views: `HUNYUAN_MAX_NUM_VIEW=4`

## Output Structure

For each object in `scene_assets.json`, the job creates:

```
gs://bucket/scenes/{sceneId}/assets/obj_{objectId}/
├── mesh.glb              # Untextured mesh from shape generation
├── model.obj             # Textured mesh (OBJ format with materials)
├── model.mtl             # Material file for OBJ
├── model.jpg             # Albedo/diffuse texture
├── model_metallic.jpg    # Metallic texture
├── model_roughness.jpg   # Roughness texture
├── model.glb             # Textured mesh (GLB format)
├── asset.glb             # Copy of model.glb (for compatibility)
└── model.usdz            # USDZ format (optional, if enabled)
```

## Integration

This job is triggered by the `hunyuan-pipeline.yaml` workflow when:
1. The assets-plan-job completes and writes `scene_assets.json`
2. The workflow detects the file and triggers this job
3. This job processes all static objects in the plan

See [workflows/hunyuan-pipeline.yaml](../workflows/hunyuan-pipeline.yaml) for workflow configuration.
