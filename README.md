# BlueprintPipeline

A production-ready pipeline for converting scene images into simulation-ready USD scenes with Isaac Lab RL training packages.

## Overview

BlueprintPipeline converts scene reconstructions (from [3D-RE-GEN](https://arxiv.org/abs/2512.17459)) into:
- **SimReady USD scenes** for Isaac Sim
- **Replicator bundles** for domain randomization
- **Isaac Lab task packages** for RL training

3D-RE-GEN is a modular, compositional pipeline for "image → sim-ready 3D reconstruction" with explicit physical constraints:
- 4-DoF ground-alignment for floor-contact objects
- Background bounding constraint for anti-penetration
- A-Q (Application-Querying) for scene-aware occlusion completion

**Reference:**
- Paper: https://arxiv.org/abs/2512.17459
- Project: https://3dregen.jdihlmann.com/
- GitHub: https://github.com/cgtuebingen/3D-RE-GEN (code pending release)

## Pipeline Architecture

```
image → 3D-RE-GEN → regen3d-job → simready-job → usd-assembly-job → replicator-job → isaac-lab-job
                          ↓              ↓              ↓                 ↓               ↓
                     manifest      physics       scene.usda       replicator/        isaac_lab/
                      layout        props                       placement_regions   env_cfg.py
                    inventory                                   variation_manifest  train_cfg.yaml
```

## Quick Start

### Local Testing (Without GCS/Cloud Run)

Genie Sim is the **default** episode-generation backend when `USE_GENIESIM` is
unset (equivalent to `USE_GENIESIM=true`). The local runner will follow the
Genie Sim job sequence, including variation asset generation and Genie Sim
export/submit/import. The default path requires the Genie Sim prerequisites
below; if you don't have them, use the lightweight path in step 2b.

- Isaac Sim installed and reachable via `ISAAC_SIM_PATH` (must include `python.sh`).
- Genie Sim repo installed at `GENIESIM_ROOT`.
- `grpcio` installed in the active Python environment.
- Genie Sim gRPC server running at `GENIESIM_HOST:GENIESIM_PORT` (default `localhost:50051`).

```bash
# 1. Generate mock 3D-RE-GEN outputs
python fixtures/generate_mock_regen3d.py --scene-id test_kitchen --output-dir ./test_scenes

# 2. Run the local pipeline (default Genie Sim path)
python tools/run_local_pipeline.py --scene-dir ./test_scenes/scenes/test_kitchen --validate
# Default steps: regen3d → scale → interactive → simready → usd → replicator →
# variation-gen → genie-sim-export → genie-sim-submit → genie-sim-import
# Optional steps: add --enable-dwm or --enable-dream2flow for extra bundles

# 2b. Lightweight local run without Genie Sim
USE_GENIESIM=false \
python tools/run_local_pipeline.py --scene-dir ./test_scenes/scenes/test_kitchen --validate
# Or explicitly request mock Genie Sim
python tools/run_local_pipeline.py --scene-dir ./test_scenes/scenes/test_kitchen --validate --mock-geniesim
# Uses BlueprintPipeline episode generation instead of Genie Sim

# 3. Run end-to-end tests
python tests/test_pipeline_e2e.py
```

## CI Security Scans

The unit-test workflow runs Bandit and Safety checks. Both scans are expected to
fail the workflow when they detect findings, so address reported vulnerabilities
before merging changes that touch Python dependencies or code.

### Local Genie Sim Fixtures

Run the local pipeline with Genie Sim enabled to generate an export bundle and submit
local data collection. Genie Sim runs in **local-only mode by default**; no API key
is required for the free/default workflow. This is the default backend when
`USE_GENIESIM` is unset (equivalent to `USE_GENIESIM=true`); see the prerequisites
in the Quick Start section above before running locally.

Prereqs for the default Genie Sim path:
- `ISAAC_SIM_PATH` points to your Isaac Sim install (must include `python.sh`).
- `GENIESIM_ROOT` points to your Genie Sim checkout.
- `grpcio` is installed in the active Python environment.
- Genie Sim gRPC server is running for local submit/import steps.

Preflight command (recommended before running Genie Sim steps):

```bash
python -m tools.geniesim_adapter.geniesim_healthcheck
```

```bash
USE_GENIESIM=true \
python tools/run_local_pipeline.py \
    --scene-dir ./test_scenes/scenes/test_kitchen \
    --use-geniesim
```

Expected outputs for fixtures (local-only):

```
scenes/{scene_id}/geniesim/
├── scene_graph.json
├── asset_index.json
├── task_config.json
├── job.json
└── merged_scene_manifest.json

scenes/{scene_id}/episodes/geniesim_{job_id}/
├── config/
│   ├── scene_manifest.json
│   └── task_config.json
└── import_manifest.json            # Produced by genie-sim-import-job
```

### Staging Genie Sim E2E (Real gRPC + Isaac Sim)

Use the staging test to validate **export → submit → import** against a real
Genie Sim gRPC server and Isaac Sim runtime. This is intended for lab staging
environments and is gated behind an explicit flag.

Prereqs:
- Isaac Sim installed and reachable via `ISAAC_SIM_PATH` (must include `python.sh`).
- Genie Sim repo installed at `GENIESIM_ROOT`.
- Genie Sim gRPC server running at `GENIESIM_HOST:GENIESIM_PORT` (default `localhost:50051`).
- Scene data includes:
  - `assets/scene_manifest.json`
  - `.usd_assembly_complete` marker
  - `.replicator_complete` marker
  - `usd/scene.usda`
  - `variation_assets/variation_assets.json`

Run (inside Isaac Sim). For the deterministic production path used in staging, set
`PIPELINE_ENV=production` and `SIMREADY_PHYSICS_MODE=deterministic`:

```bash
RUN_GENIESIM_STAGING_E2E=1 \
PIPELINE_ENV=production \
SIMREADY_PHYSICS_MODE=deterministic \
STAGING_SCENE_DIR=/mnt/gcs/scenes/<scene_id> \
GENIESIM_HOST=localhost \
GENIESIM_PORT=50051 \
ALLOW_GENIESIM_MOCK=0 \
GENIESIM_MOCK_MODE=false \
/isaac-sim/python.sh -m pytest tests/test_geniesim_staging_e2e.py -v
```

### Cloud Deployment

The pipeline runs on Google Cloud using:
- **Cloud Run Jobs** for each pipeline step
- **Cloud Workflows** for orchestration (`workflows/usd-assembly-pipeline.yaml`)
- **Cloud Storage** for scene data
- **EventArc** for triggering (on `.regen3d_complete` marker)

See the deployment runbook for step-by-step infrastructure, secrets, and workflow activation guidance:
[`docs/deployment_runbook.md`](docs/deployment_runbook.md).

### OpenUSD dependency

The USD assembly step relies on the OpenUSD Python bindings (`pxr`). The default
`usd-assembly-job` image installs these via `usd-core` (see
`usd-assembly-job/Dockerfile`). If you build a custom image or deployment guide,
ensure OpenUSD is installed so the USD assembly job can import `pxr`.

## Jobs

| Job | Purpose | Inputs | Outputs |
|-----|---------|--------|---------|
| `regen3d-job` | Adapt 3D-RE-GEN outputs | 3D-RE-GEN meshes + poses | `scene_manifest.json`, `scene_layout_scaled.json` |
| `interactive-job` | Add articulation (Particulate) | GLB meshes | URDF + segmented meshes |
| `simready-job` | Add physics properties | Manifest | `simready.usda` per object |
| `usd-assembly-job` | Build final USD scene | Manifest + layout | `scene.usda` |
| `replicator-job` | Generate domain randomization | Manifest + inventory | `placement_regions.usda`, policy scripts |
| `isaac-lab-job` | Generate RL training tasks | Manifest + USD | `env_cfg.py`, `train_cfg.yaml`, etc. |
| `dwm-preparation-job` | Generate DWM conditioning data | Manifest + USD | Egocentric videos, hand meshes, bundles |

## Output Structure

After running the pipeline, each scene has:

```
scenes/{scene_id}/
├── input/
│   └── room.jpg                    # Source image
├── regen3d/                        # 3D-RE-GEN reconstruction
│   ├── scene_info.json
│   ├── objects/
│   │   └── obj_{id}/
│   │       ├── mesh.glb
│   │       ├── pose.json
│   │       └── bounds.json
│   └── background/
├── assets/
│   ├── scene_manifest.json         # Canonical manifest
│   ├── .regen3d_complete           # Completion marker
│   └── obj_{id}/
│       └── asset.glb
├── layout/
│   └── scene_layout_scaled.json    # Scene layout with transforms
├── seg/
│   └── inventory.json              # Semantic inventory
├── usd/
│   └── scene.usda                  # Final USD scene
├── replicator/
│   ├── placement_regions.usda      # Placement surfaces
│   ├── bundle_metadata.json
│   ├── policies/                   # Replicator scripts
│   └── variation_assets/
│       └── manifest.json
├── isaac_lab/
│   ├── env_cfg.py                  # ManagerBasedEnv config
│   ├── task_{policy}.py            # Task implementation
│   ├── train_cfg.yaml              # Training hyperparameters
│   ├── randomizations.py           # Domain randomization hooks
│   └── reward_functions.py         # Reward modules
└── dwm/                            # DWM conditioning data
    ├── dwm_bundles_manifest.json   # Overall manifest
    └── {bundle_id}/
        ├── manifest.json           # Bundle metadata
        ├── static_scene_video.mp4  # Rendered static scene
        ├── hand_mesh_video.mp4     # Rendered hand meshes
        ├── camera_trajectory.json  # Camera poses
        ├── hand_trajectory.json    # Hand poses (MANO format)
        └── metadata/
            └── prompt.txt          # Text prompt for DWM
```

## Definition of Done

A scene is complete when:

- [x] `scene.usda` loads in Isaac Sim without errors
- [x] Scale is correct (countertops ~0.9m, doors ~2m)
- [x] All objects have collision proxies
- [x] Articulated objects have controllable joints
- [x] Physics simulation stable for 100+ steps
- [x] Replicator scripts execute and generate frames
- [x] Isaac Lab task imports and runs reset/step

Validate with:
```python
from tools.qa_validation import run_qa_validation
report = run_qa_validation(scene_dir=Path("scenes/scene_123"))
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `BUCKET` | GCS bucket name | - |
| `SCENE_ID` | Scene identifier | - |
| `ASSETS_PREFIX` | Assets path | `scenes/{id}/assets` |
| `LAYOUT_PREFIX` | Layout path | `scenes/{id}/layout` |
| `USD_PREFIX` | USD output path | `scenes/{id}/usd` |
| `REPLICATOR_PREFIX` | Replicator path | `scenes/{id}/replicator` |
| `REGEN3D_PREFIX` | 3D-RE-GEN path | `scenes/{id}/regen3d` |
| `ENVIRONMENT_TYPE` | Environment hint | `generic` |
| `PARTICULATE_ENDPOINT` | Particulate articulation service URL | - |
| `LLM_PROVIDER` | LLM provider (`gemini`/`openai`/`auto`) | `auto` |
| `SIMREADY_PHYSICS_MODE` | Simready physics estimation (`auto`/`gemini`/`deterministic`) | `auto` |
| `SIMREADY_ALLOW_DETERMINISTIC_PHYSICS` | Allow deterministic (LLM-free) physics when Gemini is unavailable | `false` |
| `SIMREADY_FALLBACK_MIN_COVERAGE` | Minimum coverage ratio for deterministic fallback physics (0-1) | `0.6` |
| `SIMREADY_NON_LLM_MIN_QUALITY` | Minimum quality ratio for non-LLM physics checks (0-1) | `0.85` |

## Secrets

BlueprintPipeline uses Google Secret Manager with environment variable fallbacks for development.
In production, Secret Manager is required and env var fallbacks are rejected. Configure the
following secret IDs for jobs that rely on external APIs:

| Secret ID | Env var fallback | Used by | Description |
|-----------|------------------|---------|-------------|
| `gemini-api-key` | `GEMINI_API_KEY` | `simready-job`, `episode-generation-job` | Gemini API access for physics estimation and task specification |
| `openai-api-key` | `OPENAI_API_KEY` | `episode-generation-job` | OpenAI API access for task specification |
| `anthropic-api-key` | `ANTHROPIC_API_KEY` | `episode-generation-job` | Anthropic API access for task specification |
| `particulate-api-key` | `PARTICULATE_API_KEY` | `interactive-job` | Particulate articulation service access |

In production, `simready-job` normally uses Gemini for physics estimation. You can opt into deterministic,
LLM-free physics by setting `SIMREADY_PHYSICS_MODE=deterministic` (or `SIMREADY_ALLOW_DETERMINISTIC_PHYSICS=1`);
otherwise, missing Gemini credentials cause the job to fail in production mode.

### Production modes (free vs. paid)

**Free production (deterministic, no Gemini)**:
- Required flags: `SIMREADY_PRODUCTION_MODE=1` (or `PIPELINE_ENV=production`) and
  `SIMREADY_PHYSICS_MODE=deterministic`.
- The run enforces metadata/material coverage (`SIMREADY_FALLBACK_MIN_COVERAGE`) and
  non-LLM quality checks (`SIMREADY_NON_LLM_MIN_QUALITY`) to maintain simulation fidelity.

**Paid production (Gemini-backed)**:
- Required flags: `SIMREADY_PRODUCTION_MODE=1` (or `PIPELINE_ENV=production`) and either
  `SIMREADY_PHYSICS_MODE=gemini` or `SIMREADY_PHYSICS_MODE=auto` with Gemini credentials available.
- Configure the `gemini-api-key` Secret Manager entry (production rejects env var fallbacks).

## Documentation

- [Troubleshooting](docs/troubleshooting.md)
- [Rollback Procedures](docs/rollback.md)
- [Performance Tuning](docs/performance_tuning.md)

## Testing

```bash
# Run all integration tests
python tests/test_pipeline_e2e.py

# Run specific test
python -m pytest tests/test_pipeline_e2e.py::test_full_pipeline -v

# Run local pipeline with specific steps
python tools/run_local_pipeline.py \
    --scene-dir ./scenes/test \
    --steps regen3d,simready,usd \
    --validate
```

CI enforces a unit-test coverage gate: the unit-test job publishes `coverage.xml`
from `pytest-cov` and fails if total line coverage drops below 75%. Update the
threshold in `.github/workflows/test-unit.yml` if the baseline changes.

### Staging Isaac Sim E2E (Labs pre-production)

Before production rollouts, labs should run the staging E2E harness against a
real 3D-RE-GEN reconstruction and Isaac Sim. This validates the full handoff
from reconstruction → USD → Isaac Sim loading without relying on mocks.

Staging checklist (labs pre-production):
- Verify Particulate is reachable (`PARTICULATE_ENDPOINT`) **or** run a locally hosted Particulate
  instance with `PARTICULATE_MODE=local` and an approved `PARTICULATE_LOCAL_MODEL` configured.

```bash
# Run inside an Isaac Sim environment with real reconstruction outputs
RUN_STAGING_E2E=1 \
STAGING_SCENE_DIR=/mnt/gcs/scenes/<scene_id> \
/isaac-sim/python.sh -m pytest tests/test_pipeline_e2e_staging.py -v
```

If you store scenes under a data root, you can also specify:

```bash
RUN_STAGING_E2E=1 \
STAGING_DATA_ROOT=/mnt/gcs \
STAGING_SCENE_ID=<scene_id> \
/isaac-sim/python.sh -m pytest tests/test_pipeline_e2e_staging.py -v
```

## For Robotics Labs

The pipeline produces outputs ready for integration with existing workflows:

1. **Isaac Sim Integration**: Load `scene.usda` directly
2. **RL Training**: Import `isaac_lab/` package into Isaac Lab
3. **Domain Randomization**: Use `replicator/` scripts with Omniverse Replicator

Labs must provide the Particulate service endpoint for interactive articulation
(`PARTICULATE_ENDPOINT`). In production or when `DISALLOW_PLACEHOLDER_URDF=true`,
interactive-job raises errors instead of emitting placeholder URDFs (e.g., if
Particulate is unavailable or a mesh is missing). Expect staging runs to fail
fast and emit `.interactive_failed` when that dependency is not satisfied.

Example Isaac Lab usage:
```python
from scenes.scene_123.isaac_lab import KitchenDishLoadingEnvCfg, KitchenDishLoadingTask

# Create environment
env = ManagerBasedEnv(cfg=KitchenDishLoadingEnvCfg())

# Run training
obs = env.reset()
for step in range(1000):
    action = policy(obs)
    obs, reward, done, info = env.step(action)
```

## External Dependencies

- **3D-RE-GEN**: Scene reconstruction (code pending release ~Q1 2025)
  - Paper: https://arxiv.org/abs/2512.17459
  - Project: https://3dregen.jdihlmann.com/
  - GitHub: https://github.com/cgtuebingen/3D-RE-GEN
- **Particulate**: Fast mesh articulation detection (~10s per object)
- **Isaac Sim/Lab**: NVIDIA simulation platform
- **Omniverse Replicator**: Synthetic data generation

## License

Licensed under the [MIT License](LICENSE).
