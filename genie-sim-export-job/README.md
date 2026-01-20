# Genie Sim Export Job

Cloud Run job that exports BlueprintPipeline scenes to [Genie Sim 3.0](https://github.com/AgibotTech/genie_sim) format for data generation.

## Overview

This job converts BlueprintPipeline's `scene_manifest.json` and USD scene into formats that Genie Sim 3.0 can consume for:

- **LLM-based task generation**
- **cuRobo trajectory planning** (GPU-accelerated)
- **Automated/teleop data collection**
- **VLM-based evaluation**
- **LeRobot v0.3.3 dataset export**

## Enhanced Features (DEFAULT: ENABLED)

Beyond base Genie Sim, BlueprintPipeline adds these features by default:

| Feature | Description | Default |
|---------|-------------|---------|
| **Multi-Robot** | Generate data for multiple robot types (franka, g2, ur10, gr1, fetch) | ✅ ON |
| **Bimanual** | Bimanual manipulation tasks (coordinated lift, lid opening) | ✅ ON |
| **VLA Packages** | Fine-tuning configs for OpenVLA, Pi0, SmolVLA, GR00T | ✅ ON |
| **Rich Annotations** | 2D/3D boxes, segmentation, depth GT, 6DoF poses | ✅ ON |
| **Multi-Robot Coordination** | Robot-to-robot handoffs, collaborative assembly | ✅ ON |
| **Commercial Filter** | Only include YOUR assets (exclude NC-licensed) | ✅ ON |

## Architecture

```
BlueprintPipeline                          Genie Sim 3.0
┌─────────────────┐                       ┌─────────────────┐
│ scene_manifest  │ ──── THIS JOB ────→   │ scene_graph     │
│ scene.usda      │      converts         │ asset_index     │
│ inventory.json  │                       │ task_config     │
└─────────────────┘                       │ scene_config    │
                                          └────────┬────────┘
                                                   │
                                                   ▼
                                          ┌─────────────────┐
                                          │ Task Gen (LLM)  │
                                          │ Trajectory Plan │
                                          │ Data Collection │
                                          │ VLM Evaluation  │
                                          │ LeRobot Export  │
                                          └─────────────────┘
```

## Usage

### Genie Sim Mode (Default)

Genie Sim is enabled by default. To customize robot type:

```bash
export GENIESIM_ROBOT_TYPE=franka  # or g2, ur10
```

To disable Genie Sim and use BlueprintPipeline's own episode generation:

```bash
export USE_GENIESIM=false
```

### Run Locally

```bash
cd genie-sim-export-job
export SCENE_ID=my_kitchen_001
export ASSETS_PREFIX=scenes/my_kitchen_001/assets
export GENIESIM_PREFIX=scenes/my_kitchen_001/geniesim

python export_to_geniesim.py
```

### Cloud Run

```bash
gcloud run jobs execute genie-sim-export-job \
  --set-env-vars="SCENE_ID=my_kitchen_001,ROBOT_TYPE=franka"
```

## Output Files

The job generates:

```
scenes/{scene_id}/geniesim/
├── scene_graph.json      # Nodes (objects) + Edges (spatial relations)
├── asset_index.json      # Asset metadata for RAG retrieval
├── task_config.json      # Task generation hints
├── scene_config.yaml     # Genie Sim scene configuration
├── export_manifest.json  # Export metadata + file inventory + checksums
└── usd/                  # (optional) Copied USD files
    └── scene.usda
```

## Configuration

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `SCENE_ID` | (required) | Scene identifier |
| `ROBOT_TYPE` | `franka` | Primary robot type: `franka`, `g2`, `ur10` |
| `MAX_TASKS` | `50` | Maximum suggested tasks |
| `GENERATE_EMBEDDINGS` | `false` | Generate semantic embeddings |
| `REQUIRE_EMBEDDINGS` | `false` | Fail if real embeddings cannot be generated |
| `FILTER_COMMERCIAL` | `true` | Only include commercial-use assets |
| `COPY_USD` | `true` | Copy USD files to output |

### Production Embedding Requirements

When `GENIESIM_ENV`/`BP_ENV` resolves to `production`, embedding generation is required and placeholder embeddings
are disabled. Provide an embedding provider API key (for example, `OPENAI_API_KEY` or `QWEN_API_KEY`/
`DASHSCOPE_API_KEY`) and matching embedding model environment variables so the job can complete successfully.

### Enhanced Features (NEW)

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_MULTI_ROBOT` | `true` | Generate configs for multiple robot types |
| `ENABLE_BIMANUAL` | `true` | Generate bimanual manipulation configs |
| `ENABLE_VLA_PACKAGES` | `true` | Generate VLA fine-tuning packages |
| `ENABLE_RICH_ANNOTATIONS` | `true` | Generate rich annotation configs |

## Schema Mapping

### BlueprintPipeline → Genie Sim Scene Graph

| BlueprintPipeline | Genie Sim Node |
|-------------------|----------------|
| `id` | `asset_id` |
| `category` + `description` | `semantic` |
| `dimensions_est` | `size` |
| `transform.position` | `pose.position` |
| `transform.rotation_*` | `pose.orientation` |
| `sim_role` + `affordances` | `task_tag` |
| `asset.path` | `usd_path` |

### Spatial Relations (Edges)

Edges are either extracted from `relationships` array or inferred:

| Relation | Inference Rule |
|----------|---------------|
| `on` | Vertical proximity + horizontal overlap |
| `in` | Bounds containment |
| `adjacent` | Horizontal proximity (< 15cm) |
| `aligned` | Similar orientation (< 5°) |
| `stacked` | Vertical stacking |

## Commercial Use

**Important**: When selling generated data:

1. ✅ Use only your own assets (from BlueprintPipeline)
2. ✅ Use Genie Sim tooling (MPL 2.0 license)
3. ❌ Do NOT use GenieSimAssets (CC BY-NC-SA 4.0 - non-commercial)

Set `FILTER_COMMERCIAL=true` to exclude any non-commercial assets.

## References

- [Genie Sim 3.0 Paper](https://arxiv.org/html/2601.02078v1)
- [Genie Sim GitHub](https://github.com/AgibotTech/genie_sim)
- [Integration Spec](../docs/GENIESIM_INTEGRATION.md)
