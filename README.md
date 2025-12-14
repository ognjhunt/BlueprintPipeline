# BlueprintPipeline

A production-ready pipeline for converting scene images into simulation-ready USD scenes with Isaac Lab RL training packages.

## Overview

BlueprintPipeline converts scene reconstructions (from [ZeroScene](https://arxiv.org/html/2509.23607v1)) into:
- **SimReady USD scenes** for Isaac Sim
- **Replicator bundles** for domain randomization
- **Isaac Lab task packages** for RL training

## Pipeline Architecture

```
image → ZeroScene → zeroscene-job → simready-job → usd-assembly-job → replicator-job → isaac-lab-job
                          ↓              ↓              ↓                 ↓               ↓
                     manifest      physics       scene.usda       replicator/        isaac_lab/
                      layout        props                       placement_regions   env_cfg.py
                    inventory                                   variation_manifest  train_cfg.yaml
```

## Quick Start

### Local Testing (Without GCS/Cloud Run)

```bash
# 1. Generate mock ZeroScene outputs
python fixtures/generate_mock_zeroscene.py --scene-id test_kitchen --output-dir ./test_scenes

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
- **EventArc** for triggering (on `.zeroscene_complete` marker)

## Jobs

| Job | Purpose | Inputs | Outputs |
|-----|---------|--------|---------|
| `zeroscene-job` | Adapt ZeroScene outputs | ZeroScene meshes + poses | `scene_manifest.json`, `scene_layout_scaled.json` |
| `interactive-job` | Add articulation (PhysX-Anything) | GLB meshes | URDF + segmented meshes |
| `simready-job` | Add physics properties | Manifest | `simready.usda` per object |
| `usd-assembly-job` | Build final USD scene | Manifest + layout | `scene.usda` |
| `replicator-job` | Generate domain randomization | Manifest + inventory | `placement_regions.usda`, policy scripts |
| `isaac-lab-job` | Generate RL training tasks | Manifest + USD | `env_cfg.py`, `train_cfg.yaml`, etc. |

## Output Structure

After running the pipeline, each scene has:

```
scenes/{scene_id}/
├── input/
│   └── room.jpg                    # Source image
├── zeroscene/                      # ZeroScene reconstruction
│   ├── scene_info.json
│   ├── objects/
│   │   └── obj_{id}/
│   │       ├── mesh.glb
│   │       ├── pose.json
│   │       └── bounds.json
│   └── background/
├── assets/
│   ├── scene_manifest.json         # Canonical manifest
│   ├── .zeroscene_complete         # Completion marker
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
└── isaac_lab/
    ├── env_cfg.py                  # ManagerBasedEnv config
    ├── task_{policy}.py            # Task implementation
    ├── train_cfg.yaml              # Training hyperparameters
    ├── randomizations.py           # Domain randomization hooks
    └── reward_functions.py         # Reward modules
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
| `ZEROSCENE_PREFIX` | ZeroScene path | `scenes/{id}/zeroscene` |
| `ENVIRONMENT_TYPE` | Environment hint | `generic` |
| `PHYSX_ENDPOINT` | PhysX-Anything service URL | - |
| `LLM_PROVIDER` | LLM provider (`gemini`/`openai`/`auto`) | `auto` |

## Testing

```bash
# Run all integration tests
python tests/test_pipeline_e2e.py

# Run specific test
python -m pytest tests/test_pipeline_e2e.py::test_full_pipeline -v

# Run local pipeline with specific steps
python tools/run_local_pipeline.py \
    --scene-dir ./scenes/test \
    --steps zeroscene,simready,usd \
    --validate
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

- **ZeroScene**: Scene reconstruction (code pending release)
- **PhysX-Anything**: Articulation detection ([github.com/ziangcao0312/PhysX-Anything](https://github.com/ziangcao0312/PhysX-Anything))
- **Isaac Sim/Lab**: NVIDIA simulation platform
- **Omniverse Replicator**: Synthetic data generation

## License

[Add license information]
