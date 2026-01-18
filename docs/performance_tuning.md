# Performance Tuning

Guidance for sizing resources, tuning Isaac Sim, and scaling the pipeline for higher throughput.

## Resource requirements

Resource needs vary by scene size and number of objects. Use these as starting points.

### Local development
- **CPU**: 8+ cores
- **Memory**: 32+ GB
- **Disk**: 100+ GB free (USDs, meshes, cached outputs)
- **GPU**: Optional for non-rendering steps; required for Isaac Sim rendering or Replicator

### Cloud Run Jobs
- **regen3d-job**: CPU-heavy for mesh processing; start with 4-8 vCPU, 16-32 GB RAM.
- **simready-job**: Geometry + physics proxy generation; start with 4-8 vCPU, 16-32 GB RAM.
- **usd-assembly-job**: Moderate CPU, I/O heavy; start with 2-4 vCPU, 8-16 GB RAM.
- **replicator-job**: Requires GPU for rendering; use GPU-enabled runners when running Replicator.
- **isaac-lab-job**: CPU for config generation; GPU only needed for training runs.
- **dwm-preparation-job**: GPU for rendering, with extra disk for video bundles.

## Isaac Sim tuning

### Headless execution
- Use headless mode for batch processing.
- Disable unnecessary extensions and UI services when running in Cloud Run.

### Physics stability
- Reduce `physics_dt` to improve stability if objects jitter.
- Increase solver iterations for large scenes or complex articulation.
- Validate mass and inertia tensors during `simready-job`.

### Rendering performance
- Use lower render resolution for Replicator previews.
- Minimize material complexity in assets.
- Batch render passes where possible instead of per-frame restarts.

### Asset optimization
- Simplify meshes (decimation or LODs) before physics proxy generation.
- Remove hidden or occluded geometry to reduce collision complexity.
- Normalize transforms and scales to avoid extreme values.

## Reproducing GPU tests locally

Use a host with an NVIDIA GPU and drivers installed. Verify the driver runtime with:

```bash
nvidia-smi
```

Then install dependencies and run the GPU-marked tests:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install pytest pytest-cov pytest-timeout pytest-xdist pytest-mock
pip install -r tools/requirements.txt || true
pip install -r simready-job/requirements.txt || true
pip install -r usd-assembly-job/requirements.txt || true
pip install -r replicator-job/requirements.txt || true

export PIPELINE_ENV=test
export PYTHONPATH="$(pwd)"
pytest tests/ -v -m gpu --tb=short --durations=10
```

If you are running in a container, make sure the NVIDIA Container Toolkit is configured and pass `--gpus all` to `docker run` so the tests can access the GPU.

## Pipeline scaling tips

### Parallelization
- Run jobs per scene in parallel using Cloud Workflows.
- Fan out per-object operations (e.g., physics proxies) when possible.

### Caching and reuse
- Store intermediate outputs (manifests, USDs) to avoid reprocessing.
- Use content hashes to skip unchanged assets.

### Storage layout
- Keep scene assets co-located in the same bucket/prefix to reduce latency.
- Prefer consistent path conventions for easier batch operations.

### Retry and failure isolation
- Enable retries for transient failures (GCS, network timeouts).
- Isolate heavy steps (Replicator, DWM) into separate workflows.

### Observability
- Capture per-job timing and artifact sizes in logs.
- Track per-scene throughput and queue depth to plan scaling.

## USD assembly tuning knobs

Use these environment variables to tune asset loading in `usd-assembly-job`:

- `USD_ASSET_LOAD_THREADS` (default: 4 or CPU count, whichever is lower)  
  Controls the number of threads used to prefetch asset metadata and USD references.
  Set to `1` to disable parallel prefetch.
- `USD_ASSET_METADATA_CACHE_SIZE` (default: 512)  
  LRU cache size (entries) for asset metadata reads. Set to `0` to disable caching.
- `USD_ASSET_USDZ_CACHE_SIZE` (default: 1024)  
  LRU cache size (entries) for USD reference resolution. Set to `0` to disable caching.
- `USD_ASSET_PREFETCH_CATALOG` (default: `1`)  
  Enable (`1`) or disable (`0`) catalog lookups during metadata prefetch. If disabled,
  catalog metadata will be resolved lazily per object.

## Suggested tuning checklist

- [ ] Verify CPU/memory limits match the largest scene size.
- [ ] Enable headless Isaac Sim for batch runs.
- [ ] Reduce mesh complexity before collision generation.
- [ ] Cache intermediate artifacts to avoid repeated work.
- [ ] Parallelize per-scene jobs and monitor queue depth.
