# GPU Sizing Guide

Guidance for selecting GPUs for BlueprintPipeline workloads. Use this as a
starting point for capacity planning and compare with cost estimates in
[GPU Cost Estimation](cost_tracking_estimates.md). The sizing tables below are
illustrative—update them with your own benchmarks and instance pricing.

## Workload categories

| Workload | Description | Primary bottleneck |
| --- | --- | --- |
| Genie Sim export | Convert exported Genie Sim scenes into pipeline-ready assets and manifests. | CPU + disk I/O; GPU needed only when rendering previews. |
| Isaac Sim / Replicator rendering | Generate images, annotations, or video using Isaac Sim Replicator. | GPU memory + GPU compute; VRAM spikes with large textures. |
| Training | Reinforcement learning or model training (Isaac Lab, custom pipelines). | GPU compute + VRAM; scaling depends on batch size. |

## Baseline GPU recommendations

These baselines assume headless Isaac Sim, 1080p renders, and moderate scene
complexity. Use larger GPUs if you increase resolution, scene count, or
concurrency.

| Workload | Minimum GPU | Recommended GPU class | VRAM target | Notes |
| --- | --- | --- | --- | --- |
| Genie Sim export | Optional | T4 / L4 | 8–16 GB | GPU only needed for preview renders. |
| Isaac Sim / Replicator rendering | L4 / A10 | A10 / A100 40GB | 24–40 GB | Prefer 24+ GB for multi-camera or heavy textures. |
| Training (small) | A10 | A10 / A100 40GB | 24–40 GB | Batch size limited by VRAM. |
| Training (large) | A100 80GB | A100 80GB / H100 | 80+ GB | Multi-GPU for scale-out training. |

## Example benchmark tables

### Replicator rendering (scene count vs. runtime)

| GPU | VRAM | Scenes | Resolution | Runtime (min/scene) | Peak VRAM (GB) | Concurrency |
| --- | --- | --- | --- | --- | --- | --- |
| L4 | 24 GB | 5 | 1920×1080 | 18 | 19 | 1 |
| A10 | 24 GB | 5 | 1920×1080 | 12 | 21 | 1 |
| A100 40GB | 40 GB | 5 | 1920×1080 | 8 | 28 | 2 |

### Training throughput (policy steps/sec)

| GPU | VRAM | Env count | Batch size | Steps/sec | Peak VRAM (GB) | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| A10 | 24 GB | 64 | 2048 | 3.2k | 21 | Single-GPU training. |
| A100 40GB | 40 GB | 128 | 4096 | 7.9k | 33 | Improved throughput. |
| A100 80GB | 80 GB | 256 | 8192 | 12.4k | 58 | Scales for large runs. |

### Multi-job concurrency (rendering)

| GPU | VRAM | Jobs in parallel | Per-job resolution | Total runtime (min) | Notes |
| --- | --- | --- | --- | --- | --- |
| L4 | 24 GB | 2 | 1280×720 | 42 | Near VRAM limit. |
| A10 | 24 GB | 3 | 1280×720 | 35 | Better headroom. |
| A100 40GB | 40 GB | 4 | 1920×1080 | 28 | Stable headroom. |

## Assumptions

Update these in each benchmark report so readers can compare apples-to-apples.

- **Driver/toolkit**: NVIDIA driver version, CUDA toolkit, Omniverse/Isaac Sim version.
- **Dataset size**: Scene count, average object count, and asset sizes.
- **Resolution**: Render resolution and number of cameras per scene.
- **Simulator settings**: Headless/interactive, physics step, renderer (RTX/IRay).
- **Concurrency**: Number of scenes/jobs running in parallel.

## Running and updating benchmarks

1. **Select a scenario**: choose scenes that represent small/medium/large
   workloads and freeze inputs (scene IDs + configs).
2. **Run pipeline steps**: use the local runner or batch tools to process the
   same scenario on each GPU class.
3. **Capture metrics**: runtime, GPU utilization, peak VRAM, CPU, memory, and
   storage throughput.
4. **Publish results**: update the tables below and record raw logs.

Suggested locations for results:

- **Summary tables**: `docs/benchmarks/gpu_sizing/README.md`
- **Raw logs/metrics**: `docs/benchmarks/gpu_sizing/logs/`
- **Benchmark configs**: `docs/benchmarks/gpu_sizing/configs/`

## Related references

- [Performance tuning](performance_tuning.md)
- [GPU Cost Estimation](cost_tracking_estimates.md) for translating size to spend.
- [Genie Sim GPU runtime notes](GENIE_SIM_GPU_RUNTIME.md)
