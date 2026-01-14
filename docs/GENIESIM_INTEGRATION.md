# Genie Sim 3.0 Integration Specification

## 🎉🎉🎉 ALL PREMIUM FEATURES NOW INCLUDED BY DEFAULT 🎉🎉🎉

**BREAKING CHANGE**: Previously **$345,000-$625,000 in premium upsell features** are now **FREE and included by default** in the Genie Sim 3.0 & Isaac Lab Arena pipelines!

## 🆕 NEW: Tier 4 - Validation & Media Features ($10k-$40k) - NOW FREE

### Sim2Real Validation Service ($5k-$25k/study) ✅ DEFAULT
Real-world validation trial tracking and quality guarantees:
- ✅ Real-world validation trial tracking configuration
- ✅ Sim vs real success rate comparison framework
- ✅ Transfer gap calculation with 95% confidence intervals
- ✅ Quality guarantee certificates (50%/70%/85% success rate levels)
- ✅ Failure mode comparison (sim failures vs real failures)
- ✅ Partner lab integration configuration
- ✅ Automated report generation (JSON + Markdown)
- ✅ Experiment tracking templates
- ✅ Three validation tiers: Basic (20 trials), Comprehensive (50 trials), Certification (100 trials)

Labs need this because: They want **proof** that sim data works on real robots, not just estimates!

### Audio Narration ($5k-$15k) ✅ DEFAULT
Text-to-speech narration for VLA training and accessibility:
- ✅ Text-to-speech narration synchronized with episodes
- ✅ Multi-voice presets (narrator, instructor, casual, robot)
- ✅ MP3/WAV/OGG audio output configuration
- ✅ Google Cloud TTS + local TTS fallback support
- ✅ Per-episode + combined audio generation
- ✅ Task-specific narration templates (pick_place, open_drawer, pour, generic)
- ✅ VLA audio modality training support (RT-2, PaLM-E)
- ✅ LeRobot integration (audio column, timestamps, transcripts)

Labs need this because: VLA models like RT-2 and PaLM-E can benefit from audio modality training!

## Tier 1: Essential Features ($40k-$75k) - NOW FREE

### Sim2Real Fidelity Matrix ($20k-$50k) ✅ DEFAULT
Tells you which sim aspects transfer to real robots:
- ✅ Physics fidelity scoring (friction, mass/inertia, contact dynamics, rigid body)
- ✅ Visual fidelity scoring (textures, lighting, materials, geometry)
- ✅ Sensor fidelity scoring (RGB camera, depth, proprioception, force/torque)
- ✅ Robot model fidelity (kinematics, dynamics, control, gripper)
- ✅ Domain randomization coverage analysis
- ✅ Transfer confidence score (0-100% likelihood sim→real transfer)
- ✅ Trust matrix (what to trust for training vs. validate before deployment)
- ✅ Benchmark comparison (vs RoboMimic, BridgeData v2, RLBench)

### Embodiment Transfer Analysis ($20k-$100k) ✅ DEFAULT
Answers: "Will my Franka data help train my UR10?"
- ✅ Cross-robot compatibility matrix (franka→ur10, franka→gr1, etc.)
- ✅ Kinematic similarity scoring
- ✅ Action space compatibility analysis
- ✅ Workspace overlap computation
- ✅ Predicted success rate when transferring to different robot
- ✅ Transfer efficiency score (3-5x data value multiplier for multi-robot)
- ✅ Multi-robot training strategy recommendations
- ✅ Data multiplier calculation

## Tier 2: Optimization Features ($30k-$65k) - NOW FREE

### Trajectory Optimality Analysis ($10k-$25k) ✅ DEFAULT
Ensures your training data has high-quality trajectories:
- ✅ Path length efficiency (actual vs optimal)
- ✅ Jerk analysis (smoothness scoring)
- ✅ Energy efficiency metrics
- ✅ Velocity profile analysis
- ✅ Training suitability score
- ✅ Outlier trajectory detection

### Policy Leaderboard ($20k-$40k) ✅ DEFAULT
Multi-policy comparison with statistical rigor:
- ✅ Policy rankings with confidence intervals (Wilson score, bootstrap)
- ✅ Statistical significance testing (t-test, Mann-Whitney U)
- ✅ Performance comparison across tasks and conditions
- ✅ Rank stability analysis
- ✅ Pairwise comparison matrix

### Generalization Analyzer ($15k-$35k) ✅ DEFAULT
Tells you "Do I have enough data? What should I collect next?"
- ✅ Per-object success rate analysis
- ✅ Task difficulty stratification (easy/medium/hard/expert)
- ✅ Scene variation impact analysis
- ✅ Learning curve computation
- ✅ Curriculum learning recommendations
- ✅ Data efficiency metrics

## Tier 3: Premium Features ($25k-$60k) - NOW FREE

### Tactile Sensor Simulation ($15k-$30k) ✅ DEFAULT
Research shows: 81%+ success with tactile vs ~50% vision-only!
- ✅ GelSight/GelSlim marker tracking
- ✅ DIGIT optical tactile simulation
- ✅ Magnetic tactile sensors
- ✅ Contact force maps (high-resolution 160x120 to 640x480)
- ✅ Depth/deformation maps
- ✅ Marker displacement tracking
- ✅ Contact metrics (area, centroid, force distribution)

### Language Annotations ($10k-$25k) ✅ DEFAULT
Required for VLA training (OpenVLA, Pi0, RT-2, PaLM-E):
- ✅ Template-based instruction generation
- ✅ LLM-powered variation generation (Gemini)
- ✅ Multi-style annotations (imperative, descriptive, casual, detailed, minimal)
- ✅ 10+ variations per task
- ✅ Automatic LeRobot integration
- ✅ Natural language task descriptions

## Legacy: Original Premium Analytics ($115k-$260k) - ALREADY DEFAULT

**MAJOR UPDATE**: All premium analytics features below are now **captured by default** in the Genie Sim 3.0 pipeline.

### Default Captured Analytics (No Additional Cost):

#### Per-Step Telemetry:
- ✅ Per-step rewards + reward decomposition
- ✅ Per-step collision detection (force, bodies, contact point)
- ✅ Per-step grasp events (approach→contact→grasp→lift→slip→release)
- ✅ Per-step end-effector force/torque
- ✅ Per-step joint torques

#### Failure Analysis:
- ✅ Timeout vs Collision breakdown
- ✅ Phase-level failure location (approach/grasp/lift/transport/place)
- ✅ Collision type distribution (self/table/object/gripper)
- ✅ Average collision force + locations
- ✅ Progress-at-timeout metrics

#### Grasp Analytics:
- ✅ Grasp event timeline
- ✅ Time-to-first-contact, time-to-grasp, time-to-lift, time-to-place
- ✅ Grasp force profile (max/mean/variance)
- ✅ Contact point tracking

#### Parallel Evaluation Metrics:
- ✅ GPU utilization during parallel eval
- ✅ Cross-environment variance
- ✅ Episodes/second throughput
- ✅ Statistical significance calculations

**To disable (not recommended):** Set `ENABLE_PREMIUM_ANALYTICS=false`

---

## Overview

This document specifies the integration between BlueprintPipeline and AGIBOT's Genie Sim 3.0
for hybrid synthetic data generation. BlueprintPipeline handles scene creation while Genie Sim
handles data generation (tasks, trajectories, episodes, evaluation).

**Genie Sim is the default data generation backend.** To use BlueprintPipeline's own episode
generation instead, set `USE_GENIESIM=false`.

## Mock vs Real Genie Sim (Decision Table)

| Mode | When to use | Required setup | Validation commands | Data quality + lab validation |
| --- | --- | --- | --- | --- |
| **Real Genie Sim** (Isaac Sim + gRPC) | Production data capture, lab validation, quality gating, and any dataset meant for training/eval. | Use [Runtime Bootstrap](#runtime-bootstrap-reproducible) or [Containerized Genie Sim Server](#containerized-genie-sim-server-docker-compose). Ensure Isaac Sim + Genie Sim repo installed, gRPC server running. | [Health check](#validation-checklist) + [local framework check](#validation-checklist). For lab/staging, run the [Staging E2E Test](#staging-e2e-test-real-grpc--isaac-sim). | **High-fidelity** physics + sensor outputs. **Required** for lab validation; mock is explicitly disallowed. |
| **Mock Genie Sim** (stubbed local gRPC) | Dev/test workflows, CI smoke tests, or integration checks when Isaac Sim isn’t available. | Use [Local gRPC Server Runner](#local-grpc-server-runner-fallback). Set `ALLOW_GENIESIM_MOCK=1` and `GENIESIM_MOCK_MODE=true`, or run `tools/run_local_pipeline.py --mock-geniesim`. | Run `python -m tools.geniesim_adapter.geniesim_server --health-check` and/or `python -m tools.geniesim_adapter.local_framework check` against the local stub server. | **Low-fidelity** stubbed outputs; **not suitable** for training data or lab validation. Must be re-run with real Genie Sim before release. |

## Runtime Bootstrap (Reproducible)

Use the deployment scripts under `tools/geniesim_adapter/deployment/` to install Genie Sim,
start the local gRPC server, and validate readiness:

```bash
cd tools/geniesim_adapter/deployment
./bootstrap_geniesim_runtime.sh
```

The bootstrap script performs the following:
1. Installs/clones Genie Sim (`install_geniesim.sh`).
2. Starts the Genie Sim gRPC server (unless `GENIESIM_START_SERVER=0`).
3. Runs the health check CLI (`python -m tools.geniesim_adapter.geniesim_healthcheck`).

## Containerized Genie Sim Server (Docker Compose)

For a reproducible **containerized** server that provisions Isaac Sim + Genie Sim in one
place, use `docker-compose.geniesim-server.yaml`:

```bash
export GENIESIM_HOST=0.0.0.0
export GENIESIM_PORT=50051
docker-compose -f docker-compose.geniesim-server.yaml up
```

The compose file calls `tools/geniesim_adapter/deployment/bootstrap_geniesim_runtime.sh`,
which clones Genie Sim, installs dependencies inside Isaac Sim, starts the gRPC server,
and runs a health check.

## Setup Checklist

1. **Install Isaac Sim** or pull the Isaac Sim container image.
2. **Clone Genie Sim** or allow the scripts to clone it into `GENIESIM_ROOT`.
3. **Set environment variables** (see tables below).
4. **Start the server** via:
   - `./tools/geniesim_adapter/deployment/bootstrap_geniesim_runtime.sh`, **or**
   - `docker-compose -f docker-compose.geniesim-server.yaml up`.

## Validation Checklist

1. **Preflight check** (shared by submit job + local runner):
   ```bash
   python -m tools.geniesim_adapter.geniesim_healthcheck
   ```
2. **Local availability check** (CLI wrapper around the same shared checks):
   ```bash
   python -m tools.geniesim_adapter.local_framework check
   ```

## Required Environment Variables

These are required for the local framework and health checks:

| Variable | Description | Default |
| --- | --- | --- |
| `ISAAC_SIM_PATH` | Path to Isaac Sim install (must contain `python.sh`). | `/isaac-sim` |
| `GENIESIM_ROOT` | Path to the Genie Sim repository checkout. | `/opt/geniesim` |
| `GENIESIM_HOST` | gRPC host for the Genie Sim server. | `localhost` |
| `GENIESIM_PORT` | gRPC port for the Genie Sim server. | `50051` |

## Optional Environment Variables

| Variable | Description | Default |
| --- | --- | --- |
| `GENIESIM_ALLOW_LINEAR_FALLBACK` | Allow linear interpolation fallback when cuRobo is unavailable (`1` to enable, `0` to disable). In non-production, the local framework auto-enables this fallback if cuRobo is missing and this variable is unset; in production, cuRobo is required and the framework fails fast. | Unset (auto-enable in non-production only) |
| `GENIESIM_REPO` | Genie Sim git repository URL (used by install scripts/containers). | `https://github.com/AgibotTech/genie_sim.git` |
| `GENIESIM_HEADLESS` | Run the server in headless mode (`1` to enable). | `1` |
| `GENIESIM_START_SERVER` | Auto-start the gRPC server in bootstrap/compose (`1` to enable). | `1` |
| `GENIESIM_HEALTHCHECK` | Run the health check after starting the server. | `1` |
| `GENIESIM_SERVER_LOG` | Log path for the server process. | `/tmp/geniesim_server.log` |

## Staged Validation Flow

Use this staged flow to validate Genie Sim before launching a pipeline run:

1. **Runtime bootstrap** (installs and starts server):
   ```bash
   ./tools/geniesim_adapter/deployment/bootstrap_geniesim_runtime.sh
   ```
2. **Health check** (Isaac Sim + gRPC + server readiness):
   ```bash
   python -m tools.geniesim_adapter.geniesim_healthcheck
   ```
3. **Local preflight** (called automatically by submit/run jobs):
   - `genie-sim-submit-job/submit_to_geniesim.py` runs the shared preflight helper.
   - `tools/run_local_pipeline.py` runs the same shared preflight before local runs.

## Staging E2E Test (Real gRPC + Isaac Sim)

The staging E2E test validates the **export → submit → import** flow using real gRPC
connections and an Isaac Sim runtime. It is gated behind `RUN_GENIESIM_STAGING_E2E=1`
to keep CI lightweight.

**Requirements**
- Isaac Sim installed and available at `ISAAC_SIM_PATH`.
- Genie Sim repo installed at `GENIESIM_ROOT`.
- Genie Sim gRPC server running at `GENIESIM_HOST:GENIESIM_PORT`.
- Scene directory with:
  - `assets/scene_manifest.json`
  - `.usd_assembly_complete` marker
  - `.replicator_complete` marker
  - `usd/scene.usda`
  - `variation_assets/variation_assets.json`
- Mock flags disabled: `ALLOW_GENIESIM_MOCK=0`, `GENIESIM_MOCK_MODE=false`.

**Run**

```bash
RUN_GENIESIM_STAGING_E2E=1 \
STAGING_SCENE_DIR=/mnt/gcs/scenes/<scene_id> \
GENIESIM_HOST=localhost \
GENIESIM_PORT=50051 \
ALLOW_GENIESIM_MOCK=0 \
GENIESIM_MOCK_MODE=false \
/isaac-sim/python.sh -m pytest tests/test_geniesim_staging_e2e.py -v
```

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

## Local gRPC Server Runner (Fallback)

When `GENIESIM_ROOT` is not available, BlueprintPipeline can launch a lightweight local
gRPC server for integration tests and lab connectivity checks. Use the server runner in
`tools/geniesim_adapter/geniesim_server.py`:

```bash
# Start local gRPC server (binds to 0.0.0.0:50051 by default)
python -m tools.geniesim_adapter.geniesim_server --port 50051 --log-level INFO

# Health check (uses gRPC health service when available)
python -m tools.geniesim_adapter.geniesim_server --host localhost --port 50051 --health-check
```

> Note: This local server is a stubbed implementation. Production data collection still
> requires the Isaac Sim-hosted Genie Sim server.

## Required Ports, Version, and Startup Order

**Ports**
- `50051/tcp`: Genie Sim gRPC service + health checks (default).

**Isaac Sim Version**
- Isaac Sim **2024.1.0+** is required for production data collection with Replicator.

**Startup Order (Production)**
1. Start Isaac Sim with the Genie Sim data collection server inside the Isaac Sim runtime.
2. Confirm the gRPC endpoint is reachable (`host:port`).
3. Launch BlueprintPipeline clients (e.g., `GenieSimLocalFramework`) to request observations,
   joint commands, trajectories, and recording sessions.

**Startup Order (Local Fallback)**
1. Start the local gRPC server runner (see above).
2. Point BlueprintPipeline to the local server via `GENIESIM_HOST` / `GENIESIM_PORT`.
3. Run data collection jobs that only require stubbed observations.

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

### Commercial Use Path (DEFAULT)

**By default, BlueprintPipeline exports only YOUR assets (filter_commercial_only=True).**

1. **Genie Sim Code** (MPL 2.0): Commercial use OK
2. **Your USD Scenes**: You own these
3. **Your Assets**: You own these (source="blueprintpipeline")
4. **Generated Data**: You can sell

### Non-Commercial Assets (AUTOMATICALLY FILTERED OUT)

1. **GenieSimAssets** (CC BY-NC-SA 4.0): Non-commercial only - **FILTERED OUT BY DEFAULT**
2. **External NC assets** (source="external_nc"): **FILTERED OUT BY DEFAULT**
3. If any generated data contains renders of NC-licensed assets, it inherits NC restriction

### Default Behavior

```bash
# FILTER_COMMERCIAL=true is the default
# Only your own assets (source="blueprintpipeline") are included
# GenieSimAssets and other NC sources are automatically excluded

# To include NC assets (research only, NOT for selling):
export FILTER_COMMERCIAL=false
```

### Asset Source Classification

| Source Value | Commercial OK | Included by Default |
|-------------|--------------|-------------------|
| `blueprintpipeline` | ✅ Yes | ✅ Yes |
| `blueprintpipeline_generated` | ✅ Yes | ✅ Yes |
| `geniesim_assets` | ❌ No (CC BY-NC-SA) | ❌ No |
| `external_nc` | ❌ No | ❌ No |

### Verification

The export job will warn if non-commercial assets are detected:

```
WARNING: 2 non-commercial assets included from {'geniesim_assets'}.
Generated data CANNOT be sold commercially.
Set filter_commercial_only=True to exclude these assets.
```

## Implementation Checklist

- [x] Create `tools/geniesim_adapter/` module
- [x] Implement `SceneGraphConverter` class
- [x] Implement `AssetIndexBuilder` class
- [x] Implement `TaskConfigGenerator` class
- [x] Create `genie-sim-export-job/` Cloud Run job
- [x] Update `pipeline_selector/` to route to Genie Sim (default mode)
- [x] Add Genie Sim output path to `storage_layout/`
- [x] Create integration tests (`tests/test_geniesim_adapter.py`)
- [x] Document robot configuration mapping (Franka ↔ G2)
- [x] Create `cloudbuild.yaml` for deployment
- [x] Create `workflows/genie-sim-export-pipeline.yaml`
- [x] Default to commercial-only assets (FILTER_COMMERCIAL=true)

## Robot Configuration Mapping

Genie Sim is optimized for the G2 humanoid robot. For other robots:

| BlueprintPipeline Robot | Genie Sim Equivalent | Notes |
|------------------------|---------------------|-------|
| `franka` | Custom URDF import | Use cuRobo for planning |
| `ur10` | Custom URDF import | Use cuRobo for planning |
| `fetch` | Custom URDF import | Mobile base + arm |
| — | `g2` (native) | Full dual-arm humanoid support |

## Environment Variables

For local Genie Sim server deployment scripts, see `tools/geniesim_adapter/deployment/` in this repo.

```bash
# Genie Sim connection (enabled by default)
# Set USE_GENIESIM=false to disable and use BlueprintPipeline episode generation
USE_GENIESIM=true  # (default, can be omitted)
GENIESIM_HOST=localhost
GENIESIM_PORT=50051
GENIESIM_ROOT=/opt/geniesim
ISAAC_SIM_PATH=/isaac-sim
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
