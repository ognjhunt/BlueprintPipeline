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

## Suggested tuning checklist

- [ ] Verify CPU/memory limits match the largest scene size.
- [ ] Enable headless Isaac Sim for batch runs.
- [ ] Reduce mesh complexity before collision generation.
- [ ] Cache intermediate artifacts to avoid repeated work.
- [ ] Parallelize per-scene jobs and monitor queue depth.
