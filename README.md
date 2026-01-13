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

```bash
# 1. Generate mock 3D-RE-GEN outputs
python fixtures/generate_mock_regen3d.py --scene-id test_kitchen --output-dir ./test_scenes

# 2. Run the local pipeline
python tools/run_local_pipeline.py --scene-dir ./test_scenes/scenes/test_kitchen --validate

# 3. Run end-to-end tests
python tests/test_pipeline_e2e.py
```

### Cloud Deployment

The pipeline runs on Google Cloud using:
- **Cloud Run Jobs** for each pipeline step
- **Cloud Workflows** for orchestration (`workflows/usd-assembly-pipeline.yaml`)
- **Cloud Storage** for scene data
- **EventArc** for triggering (on `.regen3d_complete` marker)

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

## Documentation

- [Troubleshooting](docs/troubleshooting.md)
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

### Staging Isaac Sim E2E (Labs pre-production)

Before production rollouts, labs should run the staging E2E harness against a
real 3D-RE-GEN reconstruction and Isaac Sim. This validates the full handoff
from reconstruction → USD → Isaac Sim loading without relying on mocks.

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

[Add license information]
