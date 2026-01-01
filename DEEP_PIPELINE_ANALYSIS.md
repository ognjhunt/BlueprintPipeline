# BlueprintPipeline - Deep End-to-End Analysis

**Generated:** 2026-01-01
**Branch:** `claude/analyze-pipeline-gaps-n761Z`

---

## Executive Summary

BlueprintPipeline is a **sophisticated, SOTA-inspired system** that converts 2D scene images into:
1. Simulation-ready USD scenes with physics
2. RL training packages (Isaac Lab compatible)
3. LeRobot-format training episodes with visual observations
4. DWM conditioning data for world model training

**Overall Status:** 🟡 **ARCHITECTURALLY COMPLETE**, but requires Isaac Sim runtime for production use.

---

## Complete Pipeline Flow

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│                           COMPLETE PIPELINE FLOW                                    │
├────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐              │
│  │  INPUT IMAGE    │ ──▶ │   3D-RE-GEN     │ ──▶ │  regen3d-job    │              │
│  │  (2D scene)     │     │  (External)     │     │  (Adapter)      │              │
│  │                 │     │  🔴 NOT AVAIL   │     │  ✅ Implemented │              │
│  └─────────────────┘     └─────────────────┘     └────────┬────────┘              │
│                                                           │                        │
│                          scene_manifest.json ◀────────────┘                        │
│                          scene_layout_scaled.json                                  │
│                          inventory.json                                            │
│                                                           │                        │
│  ┌────────────────────────────────────────────────────────▼─────────────────────┐ │
│  │                                                                               │ │
│  │  ┌─────────────────┐                      ┌─────────────────┐                │ │
│  │  │ interactive-job │  (OPTIONAL)          │  simready-job   │                │ │
│  │  │  Articulation   │ ◀─────────────────▶  │  Physics Est.   │                │ │
│  │  │  ⚠️ Particulate  │                      │  ✅ Gemini AI   │                │ │
│  │  └─────────────────┘                      └────────┬────────┘                │ │
│  │                                                    │                          │ │
│  │                                     simready.usda (physics) ◀─────────────────┘ │
│  │                                                    │                          │ │
│  └────────────────────────────────────────────────────▼─────────────────────────┘ │
│                                                                                     │
│  ┌─────────────────┐                                                               │
│  │ usd-assembly-   │  ✅ Complete USD scene with physics                          │
│  │ job             │ ──▶ scene.usda                                                │
│  └────────┬────────┘                                                               │
│           │                                                                         │
│           ├───────────────────────────────────────────────────────┐                │
│           │                                                       │                │
│           ▼                                                       ▼                │
│  ┌─────────────────┐                                    ┌─────────────────┐       │
│  │ replicator-job  │                                    │  isaac-lab-job  │       │
│  │  Domain Rand.   │                                    │  RL Package     │       │
│  │  ✅ Implemented │                                    │  ✅ Implemented │       │
│  └────────┬────────┘                                    └────────┬────────┘       │
│           │                                                       │                │
│           │ placement_regions.usda                               │ env_cfg.py     │
│           │ replicator_scripts/                                  │ task_*.py      │
│           │ variation_manifest.json                              │ train_cfg.yaml │
│           │                                                       │ rewards.py     │
│           │                                                       │ randomize.py   │
│           ▼                                                       ▼                │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │                                                                              │  │
│  │  ┌───────────────────────┐                    ┌───────────────────────┐    │  │
│  │  │  episode-generation-  │                    │  dwm-preparation-     │    │  │
│  │  │  job (SOTA Pipeline)  │                    │  job                  │    │  │
│  │  │  ✅ Full Architecture │                    │  ⚠️ Mock Renderer      │    │  │
│  │  │                       │                    │                        │    │  │
│  │  │  • TaskSpecifier      │                    │  • Camera Trajectories│    │  │
│  │  │  • AIMotionPlanner    │                    │  • Hand Motion (MANO) │    │  │
│  │  │  • CollisionPlanner   │                    │  • Scene Rendering    │    │  │
│  │  │  • TrajectorySolver   │                    │  • Bundle Packaging   │    │  │
│  │  │  • CPGenAugmenter     │                    │                        │    │  │
│  │  │  • SimValidator       │                    │                        │    │  │
│  │  │  • SensorCapture      │                    │                        │    │  │
│  │  │  • LeRobotExporter    │                    │                        │    │  │
│  │  └───────────┬───────────┘                    └───────────┬───────────┘    │  │
│  │              │                                            │                  │  │
│  │              ▼                                            ▼                  │  │
│  │  ┌───────────────────────┐                    ┌───────────────────────┐    │  │
│  │  │  LeRobot v2.0 Format  │                    │  DWM Bundles          │    │  │
│  │  │  • Parquet episodes   │                    │  • static_scene.mp4   │    │  │
│  │  │  • RGB/Depth video    │                    │  • hand_mesh.mp4      │    │  │
│  │  │  • Quality metrics    │                    │  • camera_traj.json   │    │  │
│  │  │  • Task annotations   │                    │  • prompt.txt         │    │  │
│  │  └───────────────────────┘                    └───────────────────────┘    │  │
│  │                                                                              │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                     │
└────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Component-by-Component Analysis

### 1. 3D Reconstruction (3D-RE-GEN)

**Status:** 🔴 **EXTERNAL DEPENDENCY - NOT AVAILABLE**

| Aspect | Detail |
|--------|--------|
| **What it does** | Converts 2D images → 3D meshes with 6-DoF poses |
| **Paper** | arXiv:2512.17459 |
| **Code Status** | "Pending Q1 2025" |
| **Impact** | **Complete Blocker** - Pipeline cannot start without this |

**Workaround Available:**
```bash
python fixtures/generate_mock_regen3d.py --scene-id test_kitchen --output-dir ./test_scenes
```

**What's Missing:**
- Actual mesh extraction from images
- Depth-based reconstruction
- Background mesh generation
- Real pose estimation

---

### 2. regen3d-job (Manifest Adapter)

**Status:** ✅ **IMPLEMENTED**

| Aspect | Detail |
|--------|--------|
| **Location** | `regen3d-job/regen3d_adapter_job.py` |
| **What it does** | Converts 3D-RE-GEN output → canonical manifest format |
| **Outputs** | `scene_manifest.json`, `scene_layout_scaled.json`, `inventory.json` |

**Works Well:**
- Schema validation via `manifest_schema.json`
- Object metadata extraction
- Semantic categorization

**Gaps:**
- Gemini enrichment is stub (uses fallbacks)
- Assumes 3D-RE-GEN output format that doesn't exist yet

---

### 3. interactive-job (Articulation Detection)

**Status:** ⚠️ **REQUIRES EXTERNAL SERVICE**

| Aspect | Detail |
|--------|--------|
| **Location** | `interactive-job/run_interactive_assets.py` |
| **What it does** | Detects articulation (drawers, doors, cabinets) |
| **Dependency** | **Particulate** service (~10s/object) |
| **Outputs** | URDF files for articulated objects |

**Impact if Missing:**
- Doors/drawers become static meshes
- No articulated manipulation training possible
- Limits training diversity significantly

---

### 4. simready-job (Physics Estimation)

**Status:** ✅ **IMPLEMENTED**

| Aspect | Detail |
|--------|--------|
| **Location** | `simready-job/prepare_simready_assets.py` |
| **What it does** | Estimates physics properties using Gemini AI |
| **Outputs** | `simready.usda` per object with physics |

**Properties Estimated:**
- Mass (via Gemini or density heuristics)
- Friction (static/dynamic)
- Restitution (bounciness)
- Collision proxy type (box/sphere/capsule)
- Center of mass
- Grasp regions
- Domain randomization distributions

**Works Well:**
- Gemini AI estimation (when API available)
- Fallback to material-based heuristics
- PhysX-compatible USD output

**Without Gemini:**
- Falls back to 600 kg/m³ density
- 0.6 static friction default
- Generic grasp regions

---

### 5. usd-assembly-job (USD Scene Building)

**Status:** ✅ **FULLY IMPLEMENTED**

| Aspect | Detail |
|--------|--------|
| **Location** | `usd-assembly-job/build_scene_usd.py` |
| **What it does** | Builds complete scene.usda |
| **Outputs** | `scene.usda` with all objects, physics, and hierarchy |

**Features:**
- GLB → USD conversion
- Physics wrapper generation
- Scene shell (walls, floor, ceiling)
- Proper USD reference hierarchy
- Material transfer (basic)

**This stage works reliably.**

---

### 6. replicator-job (Domain Randomization)

**Status:** ✅ **IMPLEMENTED**

| Aspect | Detail |
|--------|--------|
| **Location** | `replicator-job/generate_replicator_bundle.py` |
| **What it does** | Generates domain randomization configs |
| **Outputs** | `placement_regions.usda`, `replicator_scripts/`, `variation_manifest.json` |

**Supported Environments:** 12 types (kitchen, warehouse, grocery, etc.)
**Supported Policies:** 13 task types (pick_place, articulated_access, etc.)

**Works Well:**
- Placement region estimation
- Variation scripts for lighting/textures
- Policy-aware randomization

**Limitation:**
- Placement regions are geometric estimates (not sim-verified)

---

### 7. isaac-lab-job (RL Training Package)

**Status:** ✅ **IMPLEMENTED** (syntax-valid, not runtime-tested)

| Aspect | Detail |
|--------|--------|
| **Location** | `isaac-lab-job/generate_isaac_lab_task.py` |
| **What it does** | Generates complete Isaac Lab training package |
| **Outputs** | `env_cfg.py`, `task_*.py`, `train_cfg.yaml`, `rewards.py`, `randomizations.py` |

**Generated Structure:**
```python
isaac_lab/
├── __init__.py
├── env_cfg.py           # ManagerBasedEnv configuration
├── task_{policy}.py     # Task class with obs/reward/actions
├── train_cfg.yaml       # PPO/SAC hyperparameters
├── randomizations.py    # EventManager hooks
└── reward_functions.py  # Modular reward components
```

**What Works:**
- Syntax-valid Python generation
- Correct Isaac Lab API patterns
- Robot configs (Franka, UR10, Fetch)
- Physics profiles per policy type

**What's NOT Verified:**
- ❌ Runtime execution in actual Isaac Lab
- ❌ Observation space shapes correct
- ❌ Reward functions compute correctly
- ❌ Scene USD references valid

**Risk:** Generated code may fail at runtime with unclear errors.

---

### 8. episode-generation-job (SOTA Episode Generation) ⭐

**Status:** ✅ **FULLY ARCHITECTED** (but requires Isaac Sim for production)

| Aspect | Detail |
|--------|--------|
| **Location** | `episode-generation-job/` |
| **What it does** | Generates training episodes with visual observations |
| **Architecture** | SOTA-inspired (CP-Gen, DemoGen, AnyTask) |

**SOTA Pipeline Stages:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     EPISODE GENERATION PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. TASK SPECIFICATION (Gemini at top of stack)                             │
│     ├── TaskSpecifier: LLM generates structured task specs                  │
│     ├── Outputs: goals, constraints, keypoints, skill segments             │
│     └── Based on: DemoGen skill decomposition approach                      │
│                                                                              │
│  2. SEED EPISODE GENERATION                                                  │
│     ├── AIMotionPlanner: Generates Cartesian waypoints                      │
│     ├── TrajectorySolver: IK solving → joint trajectories                   │
│     └── Outputs: One "seed" episode per task                                │
│                                                                              │
│  3. COLLISION-AWARE PLANNING ⭐ NEW                                          │
│     ├── SceneCollisionChecker: Loads USD scene geometry                     │
│     ├── RRTPlanner: RRT-based path planning                                 │
│     ├── CollisionAwarePlanner: Replans colliding segments                   │
│     └── cuRobo integration (when available): GPU-accelerated                │
│                                                                              │
│  4. CP-GEN AUGMENTATION ⭐ NEW                                               │
│     ├── ConstraintPreservingAugmenter                                        │
│     ├── Preserves: Skill segment constraints                                │
│     ├── Replans: Free-space motions with variation                          │
│     └── Outputs: N variations per seed episode                              │
│                                                                              │
│  5. PHYSICS VALIDATION ⭐ NEW                                                │
│     ├── SimulationValidator                                                  │
│     ├── Mode 1: PhysX via Isaac Sim (real contacts, forces)                │
│     ├── Mode 2: Heuristic AABB checking (fallback)                          │
│     └── Outputs: Quality scores (0.0-1.0), pass/fail                        │
│                                                                              │
│  6. SENSOR DATA CAPTURE                                                      │
│     ├── IsaacSimSensorCapture (requires Isaac Sim)                          │
│     ├── MockSensorCapture (fallback - random noise)                         │
│     ├── Data Packs: Core/Plus/Full                                          │
│     └── Outputs: RGB, depth, segmentation, poses                            │
│                                                                              │
│  7. LEROBOT EXPORT                                                           │
│     ├── LeRobotExporter: v2.0 format                                        │
│     ├── Parquet episodes + video encoding                                   │
│     └── Quality metrics embedded                                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Files:**
| File | Purpose | Status |
|------|---------|--------|
| `generate_episodes.py` | Main orchestrator | ✅ Complete |
| `task_specifier.py` | Gemini task specification | ✅ Complete |
| `motion_planner.py` | AI-powered waypoint generation | ✅ Complete |
| `collision_aware_planner.py` | RRT + collision checking | ✅ **NEW** |
| `trajectory_solver.py` | IK solving | ✅ Complete |
| `cpgen_augmenter.py` | Constraint-preserving augmentation | ✅ **NEW** |
| `sim_validator.py` | Physics validation | ✅ **NEW** |
| `sensor_data_capture.py` | Visual observation capture | ✅ Complete |
| `isaac_sim_integration.py` | Unified Isaac Sim module | ✅ **NEW** |
| `lerobot_exporter.py` | LeRobot v2.0 export | ✅ Complete |

---

## Where the Pipeline WORKS Well ✅

### 1. Manifest Processing
- Schema validation works correctly
- Object categorization is solid
- Semantic inventory generation works

### 2. USD Scene Assembly
- GLB → USD conversion is robust
- Physics wrapper generation correct
- Scene hierarchy properly built

### 3. Domain Randomization Bundle
- Proper Replicator script generation
- Environment-aware policy selection
- Variation manifest generation

### 4. Episode Generation Architecture
- SOTA-inspired design (CP-Gen, DemoGen)
- Proper skill segment decomposition
- Constraint preservation during augmentation
- Physics validation integration
- LeRobot v2.0 export format

### 5. Motion Planning
- Waypoint generation works
- IK solving implemented
- **NEW:** Collision-aware planning with RRT
- Joint limit checking

---

## Where the Pipeline BREAKS DOWN ❌

### 🔴 CRITICAL-1: No 3D Reconstruction Source

**Impact:** Pipeline cannot start without 3D-RE-GEN or alternative.

**Current State:**
- 3D-RE-GEN code is not released
- Pipeline assumes output format that doesn't exist
- Mock generator works for testing only

**Resolution Options:**
1. Wait for 3D-RE-GEN release (Q1 2025?)
2. Integrate alternative: MASt3R, DUSt3R, or NeRF-based
3. Use pre-built scene manifests (manual creation)

---

### 🔴 CRITICAL-2: Isaac Sim Required for Production Data

**Impact:** Without Isaac Sim, sensor data is MOCK (random noise).

**What Happens Outside Isaac Sim:**
```python
# sensor_data_capture.py - what you get without Isaac Sim
frame_data.rgb_images[camera_id] = np.random.randint(0, 255, (h, w, 3))
# ^^^ This is random noise, NOT real visual data
```

**Components Affected:**
| Component | With Isaac Sim | Without Isaac Sim |
|-----------|----------------|-------------------|
| RGB Images | Real renders | Random noise |
| Depth Maps | Real depth | Mock zeros |
| Segmentation | Real masks | Empty |
| Physics Validation | PhysX simulation | AABB heuristics |
| Contact Info | Real PhysX contacts | None |

**Resolution:**
```bash
# Run pipeline inside Isaac Sim
/isaac-sim/python.sh tools/run_full_isaacsim_pipeline.py

# Or use docker-compose with Isaac Sim
docker-compose -f docker-compose.isaacsim.yaml up
```

---

### 🔴 CRITICAL-3: DWM Rendering is Mock-Only

**Impact:** DWM bundles contain placeholder videos, not real renders.

**What's Generated:**
```
dwm/
├── bundle_001/
│   ├── static_scene_video.mp4   # ❌ Placeholder frames
│   ├── hand_mesh_video.mp4      # ❌ Placeholder frames
│   └── metadata/prompt.txt      # ✅ Real prompts
```

**Why:** DWM rendering requires Isaac Sim Replicator for scene rendering.

---

### ⚠️ MAJOR-1: Generated Isaac Lab Code Not Runtime-Tested

**Risk:** Generated `env_cfg.py` may fail when actually run in Isaac Lab.

**What's Validated:**
- ✅ Python syntax
- ✅ Import structure

**What's NOT Validated:**
- ❌ Runtime execution
- ❌ Observation space shapes
- ❌ Action space dimensions
- ❌ Reward computation
- ❌ USD scene reference validity

---

### ⚠️ MAJOR-2: Articulation Requires Particulate Service

**Impact:** Without Particulate, all objects are static meshes.

**Affected Tasks:**
- `open_drawer`
- `close_drawer`
- `open_cabinet`
- `open_door`
- Any articulated manipulation

---

### ⚠️ MAJOR-3: LLM Fallback Degradation

**When Gemini API Unavailable:**
| Component | With Gemini | Without Gemini |
|-----------|-------------|----------------|
| Physics Estimation | AI-powered | Bulk density heuristics |
| Task Specification | Structured specs | Minimal specs |
| Motion Planning | LLM-enhanced | Pure heuristics |
| Grasp Regions | AI-estimated | Center of mass |

---

## Episode Generation Deep Dive

### Data Flow Through Episode Generation

```
scene_manifest.json
        │
        ▼
┌──────────────────────────┐
│  ManipulationTaskGenerator│
│  - Extracts objects       │
│  - Maps categories→tasks  │
│  - Generates task specs   │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  TaskSpecifier (Gemini)   │
│  - Goal specification     │
│  - Constraint extraction  │
│  - Skill segments         │
│  - Keypoint definition    │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  AIMotionPlanner          │
│  - Cartesian waypoints    │
│  - Phase annotations      │
│  - Gripper states         │
│  - Duration estimates     │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  CollisionAwarePlanner    │  ⭐ NEW
│  - Load scene geometry    │
│  - RRT path planning      │
│  - Collision checking     │
│  - Path smoothing         │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  TrajectorySolver         │
│  - IK solving             │
│  - Joint interpolation    │
│  - Velocity profiling     │
│  - Joint limit checking   │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  CPGenAugmenter           │  ⭐ NEW
│  - Skill segment detect   │
│  - Free-space replan      │
│  - Constraint preservation│
│  - N variations per seed  │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  SimulationValidator      │  ⭐ NEW
│  Mode 1: PhysX (Isaac Sim)│
│  - Real contacts          │
│  - Actual forces          │
│  - True collision check   │
│  Mode 2: Heuristic        │
│  - AABB collision         │
│  - Kinematic validation   │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  SensorDataCapture        │
│  Mode 1: Isaac Sim        │
│  - Real RGB renders       │
│  - Depth maps             │
│  - Segmentation masks     │
│  Mode 2: Mock             │
│  - Random noise (testing) │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  LeRobotExporter          │
│  - Parquet episodes       │
│  - Video encoding         │
│  - Quality metadata       │
│  - Task annotations       │
└──────────────────────────┘
```

### Data Pack Tiers

| Tier | Included Data | Use Case |
|------|---------------|----------|
| **Core** | RGB + robot state + actions + metadata + QC | Basic IL training |
| **Plus** | Core + depth + segmentation + 2D/3D bboxes | Perception-aware RL |
| **Full** | Plus + object poses + contacts + privileged state | Sim2real transfer |

---

## Final Output Structure

### If Pipeline Runs Successfully (with Isaac Sim)

```
scenes/{scene_id}/
├── assets/
│   ├── scene_manifest.json          # ✅ Object metadata
│   ├── .regen3d_complete            # Completion marker
│   └── obj_{id}/
│       ├── asset.glb                # Original mesh
│       └── metadata.json            # Physics metadata
│
├── layout/
│   └── scene_layout_scaled.json     # ✅ Spatial layout
│
├── seg/
│   └── inventory.json               # ✅ Semantic inventory
│
├── usd/
│   ├── scene.usda                   # ✅ Complete USD scene
│   └── obj_{id}/
│       └── simready.usda            # ✅ Physics-enabled wrapper
│
├── replicator/
│   ├── bundle_metadata.json         # ✅ Good
│   ├── placement_regions.usda       # ✅ Estimated regions
│   ├── variation_manifest.json      # ✅ Variation specs
│   └── policies/                    # ✅ Replicator scripts
│
├── isaac_lab/
│   ├── __init__.py                  # ⚠️ Syntax-valid only
│   ├── env_cfg.py                   # ⚠️ Not runtime tested
│   ├── task_{policy}.py             # ⚠️ Not runtime tested
│   ├── train_cfg.yaml               # ✅ Good
│   ├── randomizations.py            # ⚠️ Syntax-valid only
│   └── reward_functions.py          # ⚠️ Syntax-valid only
│
├── dwm/                             # ❌ MOCK WITHOUT ISAAC SIM
│   ├── dwm_bundles_manifest.json
│   └── {bundle_id}/
│       ├── static_scene_video.mp4   # ❌ Placeholder
│       ├── hand_mesh_video.mp4      # ❌ Placeholder
│       └── metadata/prompt.txt      # ✅ Real prompt
│
└── episodes/                        # ⚠️ QUALITY DEPENDS ON MODE
    ├── meta/
    │   ├── info.json                # ✅ Dataset metadata
    │   ├── stats.json               # ✅ Statistics
    │   ├── tasks.jsonl              # ✅ Task definitions
    │   └── episodes.jsonl           # ✅ Episode index
    ├── data/
    │   └── chunk-000/
    │       └── episode_*.parquet    # ⚠️ Real/mock based on mode
    ├── manifests/
    │   ├── generation_manifest.json # ✅ Full generation record
    │   └── task_coverage.json       # ✅ Task coverage report
    └── quality/
        └── validation_report.json   # ✅ Quality scores
```

### Output Quality Matrix

| Output | Isaac Sim Mode | Non-Isaac Mode |
|--------|----------------|----------------|
| `scene_manifest.json` | ✅ Good | ✅ Good |
| `scene.usda` | ✅ Good | ✅ Good |
| `simready.usda` | ✅ Good | ✅ Good (Gemini helps) |
| `replicator/` | ✅ Good | ✅ Good |
| `isaac_lab/` | ⚠️ Not tested | ⚠️ Not tested |
| `dwm/*.mp4` | ✅ Real renders | ❌ Mock frames |
| `episodes/rgb` | ✅ Real images | ❌ Random noise |
| `episodes/depth` | ✅ Real depth | ❌ Zeros |
| `episodes/trajectory` | ✅ Physics-validated | ⚠️ Heuristic only |
| Quality scores | ✅ Accurate | ⚠️ Estimates |

---

## What You Actually Get (Honest Assessment)

### With Full Setup (Isaac Sim + Gemini + 3D-RE-GEN)
- ✅ Real 3D reconstructed scenes
- ✅ Physics-accurate USD with AI-estimated properties
- ✅ Collision-free motion plans
- ✅ Physics-validated episodes
- ✅ Real visual observations
- ✅ DWM conditioning videos
- ✅ Production-ready training data

### With Partial Setup (Isaac Sim only, Mock 3D-RE-GEN)
- ⚠️ Synthetic scene from mock data
- ✅ Physics-accurate USD (with heuristics)
- ✅ Collision-free motion plans
- ✅ Physics-validated episodes
- ✅ Real visual observations
- ✅ Real DWM renders
- ⚠️ Good for pipeline testing, not production data

### Without Isaac Sim (Current Development Mode)
- ⚠️ Synthetic scene from mock data
- ⚠️ Heuristic physics only
- ✅ Collision-aware planning (RRT works)
- ⚠️ Heuristic validation only
- ❌ Random noise for visual data
- ❌ Mock DWM videos
- ❌ NOT SUITABLE for training

---

## Recommendations

### Priority 1: Enable Production Runs

1. **Set up Isaac Sim environment**
   ```bash
   # Use Isaac Sim Python
   /isaac-sim/python.sh tools/run_full_isaacsim_pipeline.py
   ```

2. **Ensure environment detection works**
   ```python
   from isaac_sim_integration import print_availability_report
   print_availability_report()
   # Should show ✅ for isaac_sim, physx, replicator
   ```

### Priority 2: Resolve 3D Input

Option A: Wait for 3D-RE-GEN release
Option B: Integrate alternative:
- MASt3R/DUSt3R (stereo depth)
- NeRFstudio (NeRF-based)
- Manual scene creation

### Priority 3: Deploy Articulation Service

- Set up Particulate service for articulation detection
- Or implement alternative articulation estimation

### Priority 4: Runtime Test Generated Code

- Add Isaac Lab integration tests
- Verify generated code runs for N steps
- Add observation shape validation

---

## Summary Table

| Stage | Status | Production Ready | Notes |
|-------|--------|------------------|-------|
| 3D Reconstruction | 🔴 | No | Needs 3D-RE-GEN or alternative |
| Manifest Adapter | ✅ | Yes | Works well |
| Articulation | ⚠️ | No | Needs Particulate |
| Physics Estimation | ✅ | Yes | Gemini or heuristics |
| USD Assembly | ✅ | Yes | Robust |
| Domain Rand | ✅ | Yes | Works well |
| Isaac Lab Package | ⚠️ | Partial | Not runtime tested |
| Episode Generation | ✅ | **Isaac Sim only** | Architecture complete |
| DWM Preparation | ⚠️ | **Isaac Sim only** | Mock without |

---

*This analysis was generated by deep inspection of the BlueprintPipeline codebase.*
