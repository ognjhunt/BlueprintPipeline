# BlueprintPipeline - Project Map for Coding Agents

> **Last Updated:** 2026-01-24
> **Purpose:** Comprehensive reference for coding agents to understand the project structure, workflows, and dependencies without reading every file.

## Table of Contents

1. [Project Overview](#project-overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Directory Structure](#directory-structure)
4. [Pipeline Flow](#pipeline-flow)
5. [Core Jobs](#core-jobs)
6. [Shared Tools & Libraries](#shared-tools--libraries)
7. [Workflows & Orchestration](#workflows--orchestration)
8. [Configuration System](#configuration-system)
9. [Data Contracts](#data-contracts)
10. [Testing Infrastructure](#testing-infrastructure)
11. [Genie Sim 3.0 Integration](#genie-sim-30-integration)
12. [Infrastructure & Deployment](#infrastructure--deployment)
13. [Quick Reference](#quick-reference)

---

## Project Overview

**BlueprintPipeline** is a production-ready, end-to-end pipeline that converts scene images into simulation-ready USD scenes with Isaac Lab RL training packages.

### Core Transformation Flow

```
Scene Image → 3D-RE-GEN Reconstruction → Physics-Ready Assets → USD Scene → RL Training Package
```

### Key Outputs

- **SimReady USD Scenes** - Physics-enabled scenes for Isaac Sim
- **Replicator Bundles** - Domain randomization configurations
- **Isaac Lab Tasks** - RL environment configurations
- **Episode Data** - Training data from simulation runs
- **VLA Fine-tuning Packages** - Data for vision-language-action models (OpenVLA, Pi0, SmolVLA, GR00T, Cosmos Policy)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT STAGE                                     │
│  ┌──────────────┐     ┌──────────────────┐     ┌─────────────────────────┐  │
│  │ Scene Image  │ ──▶ │    3D-RE-GEN     │ ──▶ │ Meshes + Poses + Bounds │  │
│  └──────────────┘     │  (External Tool) │     └─────────────────────────┘  │
│                       └──────────────────┘                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PROCESSING STAGE                                   │
│                                                                              │
│  regen3d-job ──▶ scale-job ──▶ interactive-job ──▶ simready-job             │
│       │              │               │                   │                   │
│       ▼              ▼               ▼                   ▼                   │
│   manifest       scaled          URDF +             physics                  │
│   inventory      layout      articulations          props                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            ASSEMBLY STAGE                                    │
│                                                                              │
│  usd-assembly-job ──▶ replicator-job ──▶ variation-asset-pipeline-job       │
│         │                    │                        │                      │
│         ▼                    ▼                        ▼                      │
│    scene.usda         placement_regions         variation_assets             │
│                       policy scripts                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
┌──────────────────────┐ ┌─────────────────┐ ┌─────────────────────────────┐
│    isaac-lab-job     │ │ arena-export-job│ │ genie-sim-export-job        │
│         │            │ │        │        │ │            │                │
│         ▼            │ │        ▼        │ │            ▼                │
│   env_cfg.py         │ │   Arena format  │ │ scene_graph.json            │
│   train_cfg.yaml     │ │                 │ │ task_config.json            │
└──────────────────────┘ └─────────────────┘ └─────────────────────────────┘
                                                        │
                                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          EPISODE GENERATION                                  │
│                                                                              │
│  Option A: genie-sim-submit-job ──▶ genie-sim-import-job                    │
│            (GPU, Isaac Sim, cuRobo trajectory planning)                      │
│                                                                              │
│  Option B: episode-generation-job                                            │
│            (BlueprintPipeline native generation)                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Experimental add-ons (disabled by default):**
- `dream2flow-preparation-job` for egocentric video/flow bundles.
- `dwm-preparation-job` for dexterous world model conditioning bundles.

---

## Directory Structure

```
/home/user/BlueprintPipeline/
│
├── CORE PIPELINE JOBS (20+ containerized jobs)
│   ├── regen3d-job/              # Adapt 3D-RE-GEN outputs → manifest
│   ├── scale-job/                # Scene scaling calibration
│   ├── interactive-job/          # Articulation detection (Particulate)
│   ├── simready-job/             # Physics property estimation
│   ├── usd-assembly-job/         # Build final USD scene
│   ├── replicator-job/           # Domain randomization bundles
│   ├── isaac-lab-job/            # RL task generation
│   ├── episode-generation-job/   # Physics simulation + data capture
│   ├── genie-sim-export-job/     # Export to Genie Sim format
│   ├── genie-sim-submit-job/     # Submit to Genie Sim service
│   ├── genie-sim-import-job/     # Import Genie Sim episodes
│   ├── genie-sim-gpu-job/        # GPU-accelerated Genie Sim execution
│   ├── genie-sim-local-job/      # Local Genie Sim fallback
│   ├── genie-sim-import-webhook/ # Webhook for import completion
│   ├── variation-asset-pipeline-job/  # Variation asset generation
│   ├── scene-generation-job/     # Synthetic scene generation
│   ├── arena-export-job/         # Arena integration export
│   ├── meshy-job/                # 3D mesh generation
│   ├── objects-job/              # Object detection/segmentation
│   ├── dataset-delivery-job/     # Dataset packaging
│   ├── particulate-service/      # Articulation microservice
│   └── smart-placement-engine-job/   # Placement optimization
│
├── EXPERIMENTAL / OPTIONAL JOBS (disabled by default)
│   ├── dream2flow-preparation-job/   # Dream2Flow bundle prep
│   └── dwm-preparation-job/          # DWM egocentric video generation
│
├── SHARED TOOLS (tools/)
│   ├── run_local_pipeline.py     # Main local orchestrator (2600+ lines)
│   ├── run_scene_batch.py        # Batch processor with retries
│   ├── run_full_isaacsim_pipeline.py  # Isaac Sim integration
│   │
│   ├── config/                   # Pipeline configuration
│   │   ├── pipeline_config.json  # Central config schema
│   │   ├── env.py                # Environment variable parser
│   │   └── production_mode.py    # Production enforcement
│   │
│   ├── geniesim_adapter/         # Genie Sim integration (CRITICAL)
│   │   ├── scene_graph.py        # Scene graph generation
│   │   ├── asset_index.py        # Asset index creation
│   │   ├── task_config.py        # Task configuration
│   │   ├── local_framework.py    # gRPC client implementation
│   │   ├── geniesim_grpc_pb2.py  # gRPC stubs
│   │   └── geniesim_server.py    # Local gRPC server
│   │
│   ├── llm_client/               # Unified LLM provider
│   │   └── client.py             # Gemini/OpenAI/Anthropic abstraction
│   │
│   ├── scene_manifest/           # Manifest utilities
│   │   ├── loader.py             # Manifest loading
│   │   ├── validate_manifest.py  # Schema validation
│   │   └── manifest_schema.json  # JSON schema
│   │
│   ├── quality_gates/            # Quality assurance
│   │   ├── quality_gate.py       # Gate enforcement
│   │   ├── sli_gate_runner.py    # SLI tracking
│   │   └── quality_config.json   # QA thresholds
│   │
│   ├── cosmos_policy_adapter/    # Cosmos Policy export format
│   │   ├── __init__.py           # Module entry point
│   │   ├── config.py             # Training/export configuration
│   │   ├── exporter.py           # Dataset export (Parquet + videos)
│   │   └── normalizer.py         # Action/state normalization [-1,+1]
│   │
│   ├── isaac_lab_tasks/          # Isaac Lab helpers
│   ├── inventory_enrichment/     # Metadata enrichment
│   ├── physics/                  # Physics simulation params
│   ├── validation/               # Input validation
│   ├── checkpoint/               # Pipeline checkpointing
│   ├── tracing/                  # Distributed tracing
│   └── [35+ more modules...]
│
├── WORKFLOWS (Google Cloud Workflows)
│   ├── usd-assembly-pipeline.yaml       # Main scene assembly
│   ├── episode-generation-pipeline.yaml # Episode generation
│   ├── genie-sim-export-pipeline.yaml   # Genie Sim export
│   ├── genie-sim-import-pipeline.yaml   # Episode import
│   ├── genie-sim-import-poller.yaml     # Import fallback polling
│   ├── arena-export-pipeline.yaml
│   ├── training-pipeline.yaml
│   ├── variation-assets-pipeline.yaml
│   └── [18+ more workflows...]
│
│   EXPERIMENTAL WORKFLOWS (disabled by default)
│   ├── dream2flow-preparation-pipeline.yaml
│   └── dwm-preparation-pipeline.yaml
│
├── TESTING
│   ├── tests/                    # Test suite
│   │   ├── conftest.py           # Pytest fixtures
│   │   ├── test_pipeline_e2e.py  # End-to-end tests
│   │   ├── test_geniesim_*.py    # Genie Sim tests
│   │   └── fixtures/             # Test data
│   └── fixtures/                 # Mock data generators
│       ├── generate_mock_regen3d.py
│       └── generate_mock_geniesim_local.py
│
├── INFRASTRUCTURE
│   ├── k8s/                      # Kubernetes manifests
│   │   ├── namespace-setup.yaml
│   │   ├── genie-sim-gpu-job.yaml
│   │   └── secondary-region/     # Multi-region failover
│   ├── infrastructure/           # Terraform, monitoring
│   └── scripts/                  # Deployment scripts
│
├── DOCUMENTATION
│   ├── docs/                     # Detailed documentation
│   │   ├── GENIESIM_INTEGRATION.md
│   │   ├── GENIE_SIM_GPU_RUNTIME.md
│   │   ├── deployment_runbook.md
│   │   └── troubleshooting.md
│   ├── README.md                 # Main readme
│   └── PROJECT_MAP.md            # This file
│
├── CONFIGURATION
│   ├── docker-compose.isaacsim.yaml     # Local Isaac Sim
│   ├── docker-compose.geniesim-server.yaml  # Local Genie Sim
│   ├── .env.example              # Environment template
│   ├── policy_configs/           # Policy definitions
│   └── pytest.ini                # Test configuration
│
└── SPECIALIZED MODULES
    ├── blueprint_sim/            # SimReady compilation
    │   └── recipe_compiler/      # USD physics compiler
    └── ultrashape/               # Shape optimization
```

---

## Pipeline Flow

### Default Pipeline Steps (with Genie Sim)

```python
STEPS = [
    "regen3d",           # 1. Adapt 3D-RE-GEN outputs
    "scale",             # 2. Optional scale calibration
    "interactive",       # 3. Articulation detection
    "simready",          # 4. Physics properties
    "usd",               # 5. Scene assembly
    "replicator",        # 6. Domain randomization
    "variation-gen",     # 7. Variation assets
    "genie-sim-export",  # 8. Export for Genie Sim
    "genie-sim-submit",  # 9. Submit to Genie Sim
    "genie-sim-import",  # 10. Import episodes
]
```

### Lightweight Pipeline (without Genie Sim)

```bash
USE_GENIESIM=false python tools/run_local_pipeline.py --scene-dir ./scene
```

Steps: regen3d → scale → interactive → simready → usd → replicator → isaac-lab

---

## Core Jobs

### Input Processing

| Job | Purpose | Entry Point | Key Dependencies |
|-----|---------|-------------|------------------|
| `regen3d-job` | Adapt 3D-RE-GEN outputs | `regen3d_adapter_job.py` | scene_manifest |
| `scale-job` | Scene scale calibration | `run_scale_from_layout.py` | - |
| `interactive-job` | Articulation detection | `run_interactive_assets.py` | Particulate service |
| `objects-job` | Object segmentation | `run_objects_from_layout.py` | - |

### Physics & Assembly

| Job | Purpose | Entry Point | Key Dependencies |
|-----|---------|-------------|------------------|
| `simready-job` | Physics property estimation | `prepare_simready_assets.py` | llm_client (Gemini) |
| `usd-assembly-job` | Build USD scene | `build_scene_usd.py` | pxr (OpenUSD) |
| `replicator-job` | Domain randomization | `generate_replicator_bundle.py` | llm_client |

### Training & Export

| Job | Purpose | Entry Point | Key Dependencies |
|-----|---------|-------------|------------------|
| `isaac-lab-job` | RL task generation | `generate_isaac_lab_task.py` | isaac_lab_tasks |
| `arena-export-job` | Arena integration | `arena_export_job.py` | - |
| `variation-asset-pipeline-job` | Variation assets | `run_variation_asset_pipeline.py` | - |

### Episode Generation

| Job | Purpose | Entry Point | Key Dependencies |
|-----|---------|-------------|------------------|
| `genie-sim-export-job` | Export to Genie Sim format | Multiple exporters | geniesim_adapter |
| `genie-sim-submit-job` | Submit to Genie Sim | `submit_to_geniesim.py` | gRPC |
| `genie-sim-import-job` | Import episodes | `import_from_geniesim.py` | quality_gates |
| `genie-sim-gpu-job` | GPU execution | Dockerfile-based | Isaac Sim, cuRobo |
| `episode-generation-job` | Native episode gen | `generate_episodes.py` | Isaac Sim |

### Experimental / Optional Jobs (Disabled by Default)

| Job | Purpose | Entry Point | Key Dependencies |
|-----|---------|-------------|------------------|
| `dream2flow-preparation-job` | Egocentric videos | `prepare_dream2flow_bundle.py` | - |
| `dwm-preparation-job` | DWM conditioning | `prepare_dwm_bundle.py` | - |

---

## Shared Tools & Libraries

### Critical Modules (tools/)

| Module | Purpose | Key Files |
|--------|---------|-----------|
| `config/` | Configuration management | `env.py`, `pipeline_config.json` |
| `geniesim_adapter/` | Genie Sim gRPC integration | `local_framework.py`, `scene_graph.py` |
| `cosmos_policy_adapter/` | Cosmos Policy export | `exporter.py`, `normalizer.py`, `config.py` |
| `llm_client/` | LLM provider abstraction | `client.py` (Gemini/OpenAI/Anthropic) |
| `scene_manifest/` | Manifest validation | `loader.py`, `manifest_schema.json` |
| `quality_gates/` | Quality assurance | `quality_gate.py`, `sli_gate_runner.py` |
| `validation/` | Input validation | `config_schemas.py`, `input_validation.py` |
| `checkpoint/` | Pipeline resume | Checkpoint state management |
| `physics/` | Physics parameters | `dynamic_scene.py`, `soft_body.py` |
| `isaac_lab_tasks/` | Isaac Lab helpers | Task builders |
| `tracing/` | Distributed tracing | Telemetry utilities |

### Key Entry Points

| Script | Purpose | Usage |
|--------|---------|-------|
| `run_local_pipeline.py` | Main local orchestrator | `python tools/run_local_pipeline.py --scene-dir ./scene` |
| `run_scene_batch.py` | Batch processing | `python tools/run_scene_batch.py --scenes-dir ./scenes` |
| `run_full_isaacsim_pipeline.py` | Isaac Sim pipeline | Inside Isaac Sim environment |
| `startup_validation.py` | Dependency checks | Pre-flight validation |

---

## Workflows & Orchestration

### Google Cloud Workflows

Workflows are YAML files in `/workflows/` that orchestrate Cloud Run jobs.

| Workflow | Trigger | Jobs Executed |
|----------|---------|---------------|
| `usd-assembly-pipeline.yaml` | `.regen3d_complete` marker | usd → simready → replicator → isaac-lab |
| `episode-generation-pipeline.yaml` | `.usd_complete` marker | episode-generation |
| `genie-sim-export-pipeline.yaml` | `.variation_pipeline_complete` | genie-sim-export → submit → import |
| `genie-sim-import-pipeline.yaml` | Submission complete | genie-sim-import |
| `training-pipeline.yaml` | Episodes imported | Training pipeline |

### EventArc Triggers

Workflows auto-trigger on GCS completion markers:
- `.regen3d_complete` → usd-assembly (core); dream2flow/dwm only when explicitly enabled
- `.usd_complete` → episode-generation
- `.variation_pipeline_complete` → genie-sim-export
- `.geniesim_complete` → arena-export

---

## Configuration System

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `BUCKET` | GCS bucket | - |
| `SCENE_ID` | Scene identifier | - |
| `USE_GENIESIM` | Enable Genie Sim | `true` |
| `GENIESIM_HOST` | Genie Sim host | `localhost` |
| `GENIESIM_PORT` | Genie Sim port | `50051` |
| `ISAAC_SIM_PATH` | Isaac Sim path | - |
| `GENIESIM_ROOT` | Genie Sim repo | `/opt/geniesim` |
| `LLM_PROVIDER` | LLM provider | `auto` |
| `PIPELINE_ENV` | Environment mode | `development` |
| `ENABLE_COSMOS_POLICY_EXPORT` | Cosmos Policy export | `true` |
| `COSMOS_POLICY_ACTION_CHUNK_SIZE` | Action chunk size | `16` |
| `COSMOS_POLICY_IMAGE_SIZE` | Image resize target | `256` |
| `COSMOS_POLICY_CAMERAS` | Camera list | `wrist,overhead` |

### Production Mode

```bash
PIPELINE_ENV=production
DISALLOW_PLACEHOLDER_URDF=true
```

Effects:
- Placeholder URDFs rejected
- Secret Manager required (no env var fallbacks)
- Quality gates strictly enforced
- Mock modes disabled

---

## Data Contracts

### Scene Manifest (`scene_manifest.json`)

Central data structure passing through all jobs:

```json
{
  "scene_id": "kitchen_001",
  "objects": [
    {
      "id": "obj_001",
      "category": "mug",
      "position": [0.5, 0.0, 0.8],
      "rotation": [0, 0, 0, 1],
      "bounds": {"min": [...], "max": [...]},
      "physics": {"mass": 0.3, "friction": 0.5},
      "asset_path": "assets/obj_001/mesh.glb"
    }
  ],
  "metadata": {...}
}
```

### Completion Markers (GCS)

| Marker | Meaning |
|--------|---------|
| `.regen3d_complete` | 3D-RE-GEN processing done |
| `.usd_complete` | USD assembly done |
| `.replicator_complete` | Replicator generation done |
| `.variation_pipeline_complete` | Variation assets ready |
| `.geniesim_complete` | Genie Sim episodes imported |
| `.failed` | Job failure |

### Output Structure

```
scenes/{scene_id}/
├── input/room.jpg
├── regen3d/                    # 3D-RE-GEN outputs
├── assets/
│   ├── scene_manifest.json     # Canonical manifest
│   └── obj_{id}/asset.glb
├── layout/scene_layout_scaled.json
├── usd/scene.usda              # Final USD scene
├── replicator/
│   ├── placement_regions.usda
│   └── policies/
├── isaac_lab/
│   ├── env_cfg.py
│   └── train_cfg.yaml
├── geniesim/                   # Genie Sim export
│   ├── scene_graph.json
│   ├── asset_index.json
│   └── task_config.json
├── episodes/                   # Generated episodes
│   └── geniesim_{job_id}/
│       ├── lerobot/            # LeRobot format
│       └── cosmos_policy/      # Cosmos Policy format
│           ├── meta/           # info.json, tasks, normalization
│           ├── data/           # Parquet files per episode
│           ├── videos/         # Multi-camera MP4s
│           └── config/         # Training YAML
└── vla_finetuning/             # VLA fine-tuning packages
```

---

## Testing Infrastructure

### Test Commands

```bash
# Unit tests with coverage
pytest tests/ -v --cov=. --cov-report=xml

# E2E tests
python tests/test_pipeline_e2e.py

# Genie Sim staging E2E
RUN_GENIESIM_STAGING_E2E=1 pytest tests/test_geniesim_staging_e2e.py -v

# Local pipeline validation
python tools/run_local_pipeline.py --scene-dir ./scene --validate
```

### Coverage Requirements

- Minimum: 85% line coverage (enforced in CI)
- Unit test markers: `@pytest.mark.unit`
- GPU tests: `@pytest.mark.gpu` (skipped in CI)

### Mock Fixtures

```bash
# Generate mock 3D-RE-GEN data
python fixtures/generate_mock_regen3d.py --scene-id test --output-dir ./test_scenes

# Generate mock Genie Sim data
python fixtures/generate_mock_geniesim_local.py --scene-dir ./scene
```

---

## Genie Sim 3.0 Integration

### What is Genie Sim 3.0?

**Genie Sim 3.0** is a **local-only GPU-accelerated data collection framework** by AgibotTech for robotic manipulation research.

- **GitHub:** https://github.com/AgibotTech/genie_sim
- **Protocol:** gRPC (NOT REST API)
- **Default Port:** 50051
- **Requires:** NVIDIA GPU + Isaac Sim

### Architecture

```
BlueprintPipeline                    Genie Sim 3.0 (Local)
┌─────────────────┐                 ┌─────────────────────┐
│ scene_manifest  │                 │   Isaac Sim (GPU)   │
│ scene.usda      │ ──Export──▶    │   - PhysX physics   │
│ inventory.json  │                 │   - cuRobo planner  │
└─────────────────┘                 │   - Replicator      │
                                    └──────────┬──────────┘
                                               │
                                    ┌──────────▼──────────┐
                                    │  gRPC Server :50051 │
                                    │  - Task generation  │
                                    │  - Trajectory plan  │
                                    │  - Data collection  │
                                    └─────────────────────┘
```

### Requirements to Run Genie Sim 3.0

#### Hardware Requirements

- **NVIDIA GPU:** Tesla T4, A10, or A100 (8GB+ VRAM minimum)
- **RAM:** 32GB+ recommended
- **Shared Memory:** 16GB (/dev/shm for multiprocessing)

#### Software Requirements

1. **NVIDIA Isaac Sim 2024.1.0+**
   ```bash
   # Installation path must include python.sh
   export ISAAC_SIM_PATH=/path/to/isaac-sim
   ```

2. **Genie Sim Repository**
   ```bash
   git clone https://github.com/AgibotTech/genie_sim.git /opt/geniesim
   export GENIESIM_ROOT=/opt/geniesim
   ```

3. **Python Dependencies**
   ```bash
   pip install grpcio grpcio-tools
   ```

4. **NVIDIA Container Toolkit** (for Docker)
   ```bash
   # Required for GPU access in containers
   ```

#### Environment Variables

```bash
# Core settings
export GENIESIM_HOST=localhost
export GENIESIM_PORT=50051
export GENIESIM_ROOT=/opt/geniesim
export ISAAC_SIM_PATH=/isaac-sim

# Production settings
export PIPELINE_ENV=production
export USE_GENIESIM=true
```

### Setup Steps

#### Option 1: Local Development

```bash
# 1. Install Isaac Sim (requires NVIDIA Omniverse)
# Download from: https://developer.nvidia.com/isaac-sim

# 2. Clone Genie Sim
git clone https://github.com/AgibotTech/genie_sim.git /opt/geniesim
cd /opt/geniesim
pip install -r requirements.txt

# 3. Start the gRPC server
$ISAAC_SIM_PATH/python.sh -m geniesim.server --port 50051

# 4. Verify health
python -m tools.geniesim_adapter.geniesim_healthcheck
```

#### Option 2: Docker Compose

```bash
# Start Genie Sim server container
docker-compose -f docker-compose.geniesim-server.yaml up

# Requires:
# - NVIDIA Container Toolkit installed
# - GPU with 8GB+ VRAM
# - 16GB shared memory
```

#### Option 3: GKE with GPU Nodes

```bash
# Deploy to Google Kubernetes Engine
./scripts/deploy-genie-sim-gpu-job.sh

# Requires:
# - GKE cluster with GPU node pool (nvidia-tesla-t4)
# - NGC credentials for pulling Isaac Sim image
```

### Running Genie Sim Workflow

```bash
# Full pipeline with Genie Sim
USE_GENIESIM=true python tools/run_local_pipeline.py \
    --scene-dir ./test_scenes/scenes/test_kitchen \
    --use-geniesim

# Steps executed:
# 1. genie-sim-export: Convert to Genie Sim format
# 2. genie-sim-submit: Submit to gRPC server
# 3. genie-sim-import: Import generated episodes
```

### Genie Sim Output Files

```
scenes/{scene_id}/geniesim/
├── scene_graph.json          # Scene node hierarchy
├── asset_index.json          # Asset metadata for RAG
├── task_config.json          # Task generation hints
├── scene_config.yaml         # Genie Sim configuration
└── job.json                  # Job metadata

scenes/{scene_id}/episodes/geniesim_{job_id}/
├── config/
│   ├── scene_manifest.json
│   └── task_config.json
├── episode_000/              # Per-episode data
│   ├── observations/         # RGB, depth, segmentation
│   ├── actions/              # Robot actions
│   └── metadata.json
└── import_manifest.json      # Import summary
```

### Supported Export Formats

| Format | Description | Output Path |
|--------|-------------|-------------|
| LeRobot v2/v3/0.4 | HuggingFace LeRobot (default) | `episodes/lerobot/` |
| Cosmos Policy | NVIDIA video diffusion policy | `episodes/cosmos_policy/` |
| RLDS | TensorFlow Datasets | `episodes/rlds/` |
| HDF5 | Robomimic / academic labs | `episodes/hdf5/` |
| ROS Bag | Legacy ROS systems | `episodes/rosbag/` |

### Supported VLA Models

| Model | Size | Recommended GPU |
|-------|------|-----------------|
| OpenVLA | 7B | A100-80GB |
| Pi0 | 3B+300M | A100-40GB |
| SmolVLA | 450M | RTX 4090 / A10 |
| GR00T N1.5 | N1.5 | A100-80GB |
| Cosmos Policy | 2B | H100-80GB (×8) |

### Supported Robots

| Robot | Type | DOF |
|-------|------|-----|
| Franka | Collaborative arm | 6-DOF |
| G2 | Unitree quadruped | 12-DOF |
| UR10 | Industrial cobot | 6-DOF |
| GR1 | Unitree biped | 32-DOF |
| Fetch | Mobile manipulation | 7-DOF + base |

### Without Genie Sim (Lightweight Mode)

```bash
# Skip Genie Sim entirely
USE_GENIESIM=false python tools/run_local_pipeline.py --scene-dir ./scene

# Or use mock mode for testing
python tools/run_local_pipeline.py --scene-dir ./scene --mock-geniesim
```

---

## Infrastructure & Deployment

### Deployment Options

| Option | Use Case | Requirements |
|--------|----------|--------------|
| Local Python | Development | Python 3.10+ |
| Docker Compose | Local testing | Docker + NVIDIA toolkit |
| Cloud Run | Production jobs | GCP project |
| GKE | GPU workloads | GKE cluster + GPU nodes |

### Key Deployment Files

| File | Purpose |
|------|---------|
| `scripts/deploy-episode-generation.sh` | Deploy episode gen to Cloud Run |
| `scripts/deploy-genie-sim-gpu-job.sh` | Deploy Genie Sim GPU job |
| `k8s/genie-sim-gpu-job.yaml` | GKE job manifest |
| `docker-compose.isaacsim.yaml` | Local Isaac Sim orchestration |

### Secrets (Google Secret Manager)

| Secret ID | Used By | Purpose |
|-----------|---------|---------|
| `gemini-api-key` | simready-job | Physics estimation |
| `openai-api-key` | episode-generation-job | Task specification |
| `anthropic-api-key` | episode-generation-job | Task specification |
| `particulate-api-key` | interactive-job | Articulation service |

---

## Quick Reference

### Common Commands

```bash
# Generate mock data
python fixtures/generate_mock_regen3d.py --scene-id test --output-dir ./test_scenes

# Run local pipeline
python tools/run_local_pipeline.py --scene-dir ./scene --validate

# Run specific steps
python tools/run_local_pipeline.py --scene-dir ./scene --steps regen3d,simready,usd

# Run without Genie Sim
USE_GENIESIM=false python tools/run_local_pipeline.py --scene-dir ./scene

# Health check Genie Sim
python -m tools.geniesim_adapter.geniesim_healthcheck

# Run tests
pytest tests/ -v --cov=.

# Batch processing
python tools/run_scene_batch.py --scenes-dir ./scenes --workers 4
```

### Key File Locations

| Purpose | Location |
|---------|----------|
| Main orchestrator | `tools/run_local_pipeline.py` |
| Pipeline config | `tools/config/pipeline_config.json` |
| Manifest schema | `tools/scene_manifest/manifest_schema.json` |
| Quality thresholds | `tools/quality_gates/quality_config.json` |
| Environment vars | `tools/config/ENVIRONMENT_VARIABLES.md` |
| Genie Sim adapter | `tools/geniesim_adapter/` |
| Deployment runbook | `docs/deployment_runbook.md` |

### Definition of Done (Scene Validation)

- [ ] `scene.usda` loads in Isaac Sim without errors
- [ ] Scale is correct (countertops ~0.9m, doors ~2m)
- [ ] All objects have collision proxies
- [ ] Articulated objects have controllable joints
- [ ] Physics simulation stable for 100+ steps
- [ ] Replicator scripts execute and generate frames
- [ ] Isaac Lab task imports and runs reset/step

---

## Need More Information?

- **Deployment:** `docs/deployment_runbook.md`
- **Troubleshooting:** `docs/troubleshooting.md`
- **Genie Sim Details:** `docs/GENIESIM_INTEGRATION.md`
- **API Reference:** `docs/api/README.md`
- **Environment Variables:** `tools/config/ENVIRONMENT_VARIABLES.md`
