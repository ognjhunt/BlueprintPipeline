# Genie Sim 3.0 Integration Specification

## Overview

This document specifies the integration between BlueprintPipeline and AGIBOT's Genie Sim 3.0
for hybrid synthetic data generation. BlueprintPipeline handles scene creation while Genie Sim
handles data generation (tasks, trajectories, episodes, evaluation).

**Genie Sim is the default data generation backend.** To use BlueprintPipeline's own episode
generation instead, set `USE_GENIESIM=false`.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        BlueprintPipeline                                │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                │
│  │ Scene Image  │ → │ 3D-RE-GEN    │ → │ SimReady     │                │
│  │ Generation   │   │ Reconstruction│   │ USD Assembly │                │
│  └──────────────┘   └──────────────┘   └──────────────┘                │
│         │                                     │                         │
│         ▼                                     ▼                         │
│  ┌──────────────┐                      ┌──────────────┐                │
│  │ Replicator   │                      │ YOUR USD     │                │
│  │ Bundle       │                      │ SCENE        │                │
│  └──────────────┘                      └──────────────┘                │
│                                               │                         │
└───────────────────────────────────────────────┼─────────────────────────┘
                                                │
                        ┌───────────────────────┘
                        │
                        ▼  genie-sim-export-job
┌─────────────────────────────────────────────────────────────────────────┐
│                         Genie Sim 3.0                                   │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                │
│  │ Scene Graph  │ → │ Asset        │ → │ LLM Task     │                │
│  │ Converter    │   │ Registration │   │ Generation   │                │
│  └──────────────┘   └──────────────┘   └──────────────┘                │
│                                               │                         │
│                                               ▼                         │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                │
│  │ LeRobot      │ ← │ Data         │ ← │ cuRobo       │                │
│  │ Export       │   │ Collection   │   │ Trajectory   │                │
│  └──────────────┘   └──────────────┘   └──────────────┘                │
│         │                                                               │
│         ▼                                                               │
│  ┌──────────────┐                                                       │
│  │ VLM          │                                                       │
│  │ Evaluation   │                                                       │
│  └──────────────┘                                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

## Schema Mapping

### BlueprintPipeline → Genie Sim Scene Graph

Genie Sim expects a hierarchical Scene Graph with:
- **Nodes**: Objects encoded with `asset_id`, `semantic`, `size`, `pose`, `task_tag`
- **Edges**: Spatial relations: `on`, `in`, `adjacent`, `aligned`, `stacked`

#### Node Mapping (SceneObject → GenieSim Node)

| BlueprintPipeline Field | GenieSim Node Field | Transformation |
|------------------------|---------------------|----------------|
| `id` | `asset_id` | Direct mapping |
| `category` + `description` | `semantic` | Concatenate for semantic description |
| `dimensions_est.{width,depth,height}` | `size` | Direct as [w, d, h] array |
| `transform.position.{x,y,z}` | `pose.position` | Direct as [x, y, z] array |
| `transform.rotation_quaternion.{w,x,y,z}` | `pose.orientation` | Direct as [w, x, y, z] array |
| `transform.rotation_euler.{roll,pitch,yaw}` | `pose.orientation` | Convert to quaternion |
| `sim_role` + `semantics.affordances` | `task_tag` | Map to GenieSim task categories |
| `asset.path` | `usd_path` | Convert to absolute or relative USD path |

#### Edge Mapping (relationships → GenieSim Edges)

| BlueprintPipeline Relationship | GenieSim Edge Type | Notes |
|-------------------------------|-------------------|-------|
| `on_top_of` | `on` | Surface contact |
| `inside` | `in` | Containment |
| `next_to` | `adjacent` | Proximity |
| `aligned_with` | `aligned` | Orientation alignment |
| `stacked_on` | `stacked` | Vertical stacking |

#### Edge Inference (when relationships not explicit)

If `relationships` array is empty, infer edges from:
1. **Vertical proximity** + `is_floor_contact=false` → `on` edge
2. **Containment check** (bounds inside bounds) → `in` edge
3. **Horizontal proximity** (< 0.1m) → `adjacent` edge
4. **Similar rotation** (< 5° difference) → `aligned` edge

### BlueprintPipeline → Genie Sim Asset Index

Genie Sim Asset Index stores per-asset:
- USD paths
- Collision hulls
- Mass properties
- Texture variants
- Semantic descriptions (for RAG retrieval)
- 2048-dim embeddings (QWEN text-embedding-v4)

#### Asset Mapping

| BlueprintPipeline Field | GenieSim Asset Index Field | Transformation |
|------------------------|---------------------------|----------------|
| `asset.path` | `usd_path` | Ensure .usdz/.usda extension |
| `physics.collision_shape` | `collision_hull` | Use existing or generate convex hull |
| `physics.mass` | `mass` | Direct |
| `physics.friction` | `material.friction` | Direct |
| `physics.restitution` | `material.restitution` | Direct |
| `asset.variants` | `texture_variants` | Map variant names to paths |
| `category` + `description` | `semantic_description` | Generate rich description for RAG |
| — | `embedding` | Generate via QWEN text-embedding-v4 |

### Task Tag Mapping

| BlueprintPipeline sim_role | BlueprintPipeline Affordances | GenieSim task_tag |
|---------------------------|------------------------------|-------------------|
| `manipulable_object` | `Graspable` | `pick`, `place` |
| `manipulable_object` | `Stackable` | `stack` |
| `manipulable_object` | `Pourable` | `pour` |
| `articulated_furniture` | `Openable` (revolute) | `open`, `close` |
| `articulated_furniture` | `Openable` (prismatic) | `pull`, `push` |
| `articulated_appliance` | `Turnable` | `turn` |
| `articulated_appliance` | `Pressable` | `press` |
| `interactive` | — | `interact` |
| `static` | `Supportable` | `place_on` |

## Data Flow

### Input: BlueprintPipeline Scene

```
scenes/{scene_id}/
├── assets/
│   └── scene_manifest.json    # Canonical manifest
├── seg/
│   └── inventory.json         # Object inventory with affordances
├── layout/
│   └── scene_layout_scaled.json  # Camera + room bounds
├── usd/
│   └── scene.usda             # Assembled USD scene
└── replicator/
    └── placement_regions.usda # Domain randomization regions
```

### Output: Genie Sim Export Package

```
scenes/{scene_id}/geniesim/
├── scene_graph.json           # Converted scene graph (nodes + edges)
├── asset_index.json           # Asset registration for RAG
├── task_config.json           # Task generation hints
├── scene_config.yaml          # Genie Sim scene configuration
└── usd/
    └── scene_geniesim.usda    # USD with Genie Sim conventions
```

### Output: Genie Sim Generated Data (from Genie Sim)

```
scenes/{scene_id}/episodes/
├── meta/
│   ├── info.json              # LeRobot v0.3.3 metadata
│   ├── episodes.jsonl         # Episode manifest
│   └── tasks.jsonl            # Task descriptions
├── data/
│   └── chunk-000/
│       └── episode_*.parquet  # Trajectory data
└── videos/
    └── chunk-000/
        └── observation.images.*/
            └── episode_*.mp4  # Visual observations
```

## Jobs Modified/Replaced

### Jobs REMOVED (replaced by Genie Sim)

| Job | Reason |
|-----|--------|
| `episode-generation-job` | Genie Sim handles task generation, trajectory planning, data collection |
| `isaac-lab-job` | Genie Sim includes task evaluation via VLM |

### Jobs ADDED

| Job | Purpose |
|-----|---------|
| `genie-sim-export-job` | Convert BlueprintPipeline manifest → Genie Sim format |

### Jobs MODIFIED

| Job | Changes |
|-----|---------|
| `usd-assembly-job` | Add Genie Sim-compatible metadata to USD |
| `simready-job` | Ensure physics properties compatible with Genie Sim |

### Jobs UNCHANGED

| Job | Notes |
|-----|-------|
| `scene-generation-job` | Scene image generation unchanged |
| `regen3d-job` | 3D reconstruction unchanged |
| `interactive-job` | Articulation detection unchanged |
| `replicator-job` | Domain randomization still useful |
| `dwm-preparation-job` | DWM data generation is unique to BlueprintPipeline |
| `dream2flow-preparation-job` | Dream2Flow unique to BlueprintPipeline |

## Genie Sim API Integration

### Scene Graph JSON Schema

```json
{
  "scene_id": "string",
  "coordinate_system": "y_up | z_up",
  "meters_per_unit": 1.0,
  "nodes": [
    {
      "asset_id": "string",
      "semantic": "string",
      "size": [0.1, 0.1, 0.1],
      "pose": {
        "position": [0.0, 0.0, 0.0],
        "orientation": [1.0, 0.0, 0.0, 0.0]
      },
      "task_tag": ["pick", "place"],
      "usd_path": "string",
      "properties": {
        "mass": 0.5,
        "friction": 0.5,
        "restitution": 0.1
      }
    }
  ],
  "edges": [
    {
      "source": "object_id_1",
      "target": "object_id_2",
      "relation": "on | in | adjacent | aligned | stacked"
    }
  ]
}
```

### Asset Index JSON Schema

```json
{
  "assets": [
    {
      "asset_id": "string",
      "usd_path": "string",
      "collision_hull_path": "string",
      "mass": 0.5,
      "material": {
        "friction": 0.5,
        "restitution": 0.1
      },
      "texture_variants": {
        "default": "path/to/texture.png",
        "worn": "path/to/texture_worn.png"
      },
      "semantic_description": "A white ceramic coffee mug with handle",
      "categories": ["kitchen", "container", "graspable"],
      "embedding": [0.1, 0.2, ...]
    }
  ]
}
```

### Task Configuration Schema

```json
{
  "scene_id": "string",
  "environment_type": "kitchen | warehouse | office | ...",
  "suggested_tasks": [
    {
      "task_type": "pick_place",
      "target_object": "mug_001",
      "goal_region": "countertop",
      "difficulty": "easy | medium | hard"
    }
  ],
  "robot_config": {
    "type": "g2 | franka | ur10",
    "base_position": [0.0, 0.0, 0.0],
    "workspace_bounds": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
  }
}
```

## Licensing Considerations

### Commercial Use Path

1. **Genie Sim Code** (MPL 2.0): Commercial use OK
2. **Your USD Scenes**: You own these
3. **Your Assets**: You own these
4. **Generated Data**: You can sell

### Non-Commercial Assets (DO NOT USE for selling data)

1. **GenieSimAssets** (CC BY-NC-SA 4.0): Non-commercial only
2. If any generated data contains renders of NC-licensed assets, it inherits NC restriction

### Recommended Practice

```python
# In asset registration, flag commercial status
asset_entry = {
    "asset_id": "my_mug_001",
    "commercial_ok": True,  # Your asset
    "license": "proprietary",
    "source": "blueprintpipeline_generated"
}

# Never mix with NC assets for sellable data
assert all(a["commercial_ok"] for a in scene_assets)
```

## Implementation Checklist

- [ ] Create `tools/geniesim_adapter/` module
- [ ] Implement `SceneGraphConverter` class
- [ ] Implement `AssetIndexBuilder` class
- [ ] Implement `TaskConfigGenerator` class
- [ ] Create `genie-sim-export-job/` Cloud Run job
- [ ] Update `pipeline_selector/` to route to Genie Sim
- [ ] Add Genie Sim output path to `storage_layout/`
- [ ] Create integration tests
- [ ] Document robot configuration mapping (Franka ↔ G2)

## Robot Configuration Mapping

Genie Sim is optimized for the G2 humanoid robot. For other robots:

| BlueprintPipeline Robot | Genie Sim Equivalent | Notes |
|------------------------|---------------------|-------|
| `franka` | Custom URDF import | Use cuRobo for planning |
| `ur10` | Custom URDF import | Use cuRobo for planning |
| `fetch` | Custom URDF import | Mobile base + arm |
| — | `g2` (native) | Full dual-arm humanoid support |

## Environment Variables

```bash
# Genie Sim connection (enabled by default)
# Set USE_GENIESIM=false to disable and use BlueprintPipeline episode generation
USE_GENIESIM=true  # (default, can be omitted)
GENIESIM_ASSETS_PATH=/path/to/geniesim/assets
GENIESIM_CHROMADB_PATH=/path/to/chromadb
GENIESIM_EMBEDDING_MODEL=qwen-text-embedding-v4

# Robot selection
GENIESIM_ROBOT_TYPE=franka  # or g2, ur10, custom
GENIESIM_ROBOT_URDF=/path/to/robot.urdf

# Data generation
GENIESIM_EPISODES_PER_TASK=100
GENIESIM_USE_TELEOP=false
GENIESIM_USE_AUTOMATED=true

# Evaluation
GENIESIM_EVAL_ENABLED=true
GENIESIM_VLM_ENDPOINT=http://localhost:8080/vlm
```

## References

- [Genie Sim 3.0 Paper](https://arxiv.org/html/2601.02078v1)
- [Genie Sim GitHub](https://github.com/AgibotTech/genie_sim)
- [GenieSimAssets (Hugging Face)](https://huggingface.co/datasets/agibot-world/GenieSimAssets)
- [GenieSim3.0-Dataset (ModelScope)](https://modelscope.cn/datasets/agibot_world/GenieSim3.0-Dataset)
