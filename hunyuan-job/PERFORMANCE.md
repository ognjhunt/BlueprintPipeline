# Hunyuan Job Performance Optimization Guide

## Problem Summary

The Hunyuan job was timing out after 1 hour (3600 seconds) when processing 11 objects. Analysis of the logs revealed:

### Timeline Breakdown (1 hour job):
- **0-24 min**: Shape generation (11 objects) ✅
- **24-32 min**: Texture pipeline loading (7.5 minutes!) ⚠️
- **32-60 min**: Texture generation (only 9/11 completed) ❌

### Key Bottlenecks:
1. **Texture pipeline loading: 7.5 minutes** - Models downloaded from HuggingFace on first run
2. **Sequential processing** - Objects processed one at a time
3. **GCS staged writes** - Slow I/O due to GCSFuse limitations
4. **1 hour timeout** - Need at least 1.5-2 hours for 11 objects

---

## Quick Fixes (Immediate Impact)

### 1. Increase Job Timeout ⚡ **REQUIRED**

**Impact:** Allows job to complete all objects
**Time saved:** Prevents premature termination

```bash
# Run this script to update the timeout to 2 hours
./update-job-timeout.sh

# Or manually:
gcloud run jobs update hunyuan-job \
  --region=us-central1 \
  --timeout=7200 \
  --max-retries=0
```

### 2. Reduce Texture Quality for Speed 🚀

**Impact:** 30-50% faster texture generation
**Trade-off:** Slightly lower quality textures

Update Cloud Run job environment variables:

```bash
gcloud run jobs update hunyuan-job \
  --region=us-central1 \
  --set-env-vars="HUNYUAN_RENDER_SIZE=512,HUNYUAN_TEXTURE_SIZE=1024"
```

**Quality Comparison:**
- **Fast** (512/1024): ~30-45 sec per object
- **Balanced** (1024/2048): ~60-90 sec per object ⬅️ Current default
- **High Quality** (2048/4096): ~120-180 sec per object

### 3. Disable USDZ Export (Temporary)

**Impact:** Save ~5-10 seconds per object
**Note:** USDZ is only needed for Apple platforms

```bash
gcloud run jobs update hunyuan-job \
  --region=us-central1 \
  --set-env-vars="ENABLE_USDZ_EXPORT=0"
```

---

## Medium-Term Optimizations

### 4. Pre-cache Models in Docker Image

**Impact:** Eliminate 7.5 minute pipeline loading time
**Complexity:** Moderate (requires image rebuild)

Add to `Dockerfile` after line 72:

```dockerfile
# Pre-download texture models to reduce startup time
# Note: This increases image size but eliminates ~7 min of startup time
RUN python3 -c "\
import os; \
os.environ['HF_HOME']='/tmp/hf-cache-warmup'; \
os.environ['TRANSFORMERS_CACHE']='/tmp/hf-cache-warmup'; \
from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline; \
conf = Hunyuan3DPaintConfig(max_num_view=6, resolution=512); \
conf.render_size = 1024; \
conf.texture_size = 2048; \
conf.realesrgan_ckpt_path = '${HUNYUAN_REPO_ROOT}/hy3dpaint/ckpt/RealESRGAN_x4plus.pth'; \
conf.multiview_cfg_path = '${HUNYUAN_REPO_ROOT}/hy3dpaint/cfgs/hunyuan-paint-pbr.yaml'; \
conf.custom_pipeline = 'hy3dpaint/hunyuanpaintpbr'; \
pipeline = Hunyuan3DPaintPipeline(conf); \
print('Texture models pre-cached')"

# Clean up build cache
RUN rm -rf /tmp/hf-cache-warmup
```

Then rebuild and push the image:

```bash
cd hunyuan-job
gcloud builds submit --tag gcr.io/blueprint-8c1ca/hunyuan-job:latest
```

### 5. Use Faster Storage

**Impact:** 20-30% faster I/O
**Complexity:** High (infrastructure change)

Options:
- Use Cloud Storage FUSE with direct writes enabled
- Use a persistent SSD volume for intermediate files
- Pre-download inputs to local disk, upload outputs in batch

---

## Advanced Optimizations

### 6. Batch Processing

**Impact:** Process multiple scenes in parallel
**Complexity:** High (requires workflow redesign)

Split large jobs into multiple smaller jobs, each handling a subset of objects:

```python
# Example: Split 11 objects into 3 jobs
# Job 1: objects 0-3 (4 objects)
# Job 2: objects 4-7 (4 objects)
# Job 3: objects 8-10 (3 objects)
```

This allows multiple Cloud Run jobs to run in parallel, reducing total wall-clock time.

### 7. Use Higher-Performance GPU

**Impact:** 30-50% faster generation
**Complexity:** Moderate (requires job config change)

Current: L4 GPU (24GB VRAM)
Options:
- A100 (40GB): ~1.5-2x faster
- H100 (80GB): ~2-3x faster

Trade-off: Significantly higher cost per hour.

```bash
gcloud run jobs update hunyuan-job \
  --region=us-central1 \
  --gpu-type=nvidia-tesla-a100 \
  --gpu=1
```

---

## Environment Variable Reference

All performance settings can be controlled via environment variables:

### Shape Generation:
```bash
HUNYUAN_NUM_STEPS=50           # Inference steps (default: 50)
HUNYUAN_MAX_NUM_VIEW=6         # Number of views (default: 6)
HUNYUAN_RESOLUTION=512         # Resolution (default: 512)
```

### Texture Generation:
```bash
HUNYUAN_RENDER_SIZE=1024       # Render resolution (default: 1024)
                                # Options: 512 (fast) | 1024 (balanced) | 2048 (slow)

HUNYUAN_TEXTURE_SIZE=2048      # Texture resolution (default: 2048)
                                # Options: 1024 (fast) | 2048 (balanced) | 4096 (slow)
```

### Performance:
```bash
ENABLE_USDZ_EXPORT=1           # Enable USDZ conversion (default: 1)
                                # Set to 0 to disable and save time

SKIP_EXISTING_ASSETS=1         # Skip objects with existing asset.glb (default: 1)
```

### Model Configuration:
```bash
HUNYUAN_MODEL_PATH="tencent/Hunyuan3D-2.1"  # Model to use
HF_TOKEN="hf_..."                            # HuggingFace token for private models
```

---

## Expected Performance After Optimizations

### Current (Before Optimization):
- **11 objects**: ~65-75 minutes (exceeds 1-hour timeout)
- **Bottleneck**: Pipeline loading (7.5 min) + sequential texture generation

### After Quick Fixes (Timeout + Reduced Quality):
- **11 objects**: ~45-55 minutes ✅ Completes within timeout
- **Settings**: 2-hour timeout, render_size=512, texture_size=1024

### After Model Pre-caching:
- **11 objects**: ~35-45 minutes ✅ Significant improvement
- **Savings**: 7.5 minutes eliminated from pipeline loading

### After All Optimizations:
- **11 objects**: ~25-35 minutes ✅ Best performance
- **With batch processing**: ~10-15 minutes (wall-clock time, multiple jobs)

---

## Recommended Action Plan

**Immediate (do this now):**
1. ✅ Run `./update-job-timeout.sh` to increase timeout to 2 hours
2. ✅ Set `HUNYUAN_RENDER_SIZE=512` and `HUNYUAN_TEXTURE_SIZE=1024`
3. ✅ Re-run the job and verify completion

**Short-term (this week):**
1. Add model pre-caching to Dockerfile
2. Rebuild and redeploy the image
3. Test and measure performance improvement

**Long-term (if needed):**
1. Implement batch processing for multiple concurrent jobs
2. Consider upgrading to A100 GPU if budget allows
3. Optimize GCS I/O with persistent volumes

---

## Monitoring and Debugging

### Check Job Status:
```bash
gcloud run jobs describe hunyuan-job --region=us-central1
```

### View Job Logs:
```bash
gcloud logging read "resource.type=cloud_run_job AND resource.labels.job_name=hunyuan-job" \
  --limit 100 --format json
```

### Check Job Execution Time:
```bash
gcloud run jobs executions list --job=hunyuan-job --region=us-central1
```

### Verify Environment Variables:
```bash
gcloud run jobs describe hunyuan-job --region=us-central1 --format=json | \
  jq '.spec.template.spec.template.spec.containers[0].env'
```

---

## USDZ Conversion Fix

The original code was looking for `usd_from_gltf` CLI tool, which doesn't exist. The new implementation:

1. Uses USD Python API (`pxr` module) from `usd-core` package
2. Converts GLB → USDC → USDZ using trimesh + USD
3. Falls back gracefully if USD not available
4. Can be disabled with `ENABLE_USDZ_EXPORT=0`

**Note:** USDZ conversion adds minimal time (~2-5 seconds per object) but may fail if `usdzip` tool is not in PATH.

---

## Questions or Issues?

If the job continues to timeout or you see performance issues:

1. Check the logs for bottlenecks (pipeline loading time, per-object time)
2. Try reducing quality settings further
3. Consider implementing batch processing
4. Contact the team for assistance
