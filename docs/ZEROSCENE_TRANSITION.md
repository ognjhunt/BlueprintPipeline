# ZeroScene Transition Guide

This document describes the transition from the current reconstruction pipeline to a ZeroScene-first approach.

## Overview

[ZeroScene](https://arxiv.org/html/2509.23607v1) provides an improved reconstruction pipeline that covers:
- Instance segmentation + depth extraction
- Object pose optimization using 3D/2D projection losses
- Foreground + background mesh reconstruction
- PBR material estimation for improved rendering realism
- Triangle mesh outputs for each object

## Jobs Status After ZeroScene

### Jobs to be DEPRECATED (Replaced by ZeroScene)

These jobs will be replaced by ZeroScene when it becomes available:

| Job | Current Purpose | ZeroScene Replacement | Status |
|-----|----------------|----------------------|--------|
| `seg-job` | Segmentation + inventory | ZeroScene segmentation | **DEPRECATED** |
| `multiview-job` | Object isolation/generative views | ZeroScene foreground/background | **DEPRECATED** |
| `scene-da3-job` | Depth/point cloud extraction | ZeroScene depth extraction | **DEPRECATED** |
| `layout-job` | Layout reconstruction | ZeroScene pose optimization | **DEPRECATED** |
| `sam3d-job` | 3D mesh generation | ZeroScene mesh reconstruction | **DEPRECATED** |
| `hunyuan-job` | Texture/refinement | ZeroScene PBR materials | **DEPRECATED** |

**Note:** Keep these jobs as a fallback while ZeroScene is not fully operational. The Gemini pipeline remains a coherent ingestion path.

### Jobs to KEEP (Still Required)

These jobs are still necessary because ZeroScene does not provide SimReady Isaac Sim training packages:

| Job | Purpose | Notes |
|-----|---------|-------|
| `zeroscene-job` | **NEW** - Adapter for ZeroScene outputs | Converts ZeroScene → manifest + layout |
| `interactive-job` | Articulation bridge (PhysX-Anything) | Adds joints to doors/drawers |
| `simready-job` | Physics + manipulation hints | Adds mass/friction/graspability |
| `usd-assembly-job` | Convert + assemble scene.usda | Final USD assembly |
| `replicator-job` | Policy scripts + placement regions | Domain randomization |
| `variation-gen-job` | Domain randomization assets | Generates clutter objects |
| `isaac-lab-job` | **NEW** - Isaac Lab task generation | RL training packages |

## New Integration Points

### 1. ZeroScene Adapter Job (`zeroscene-job`)

The adapter converts ZeroScene outputs to BlueprintPipeline format:

```
zeroscene_output/
├── scene_info.json
├── objects/
│   ├── obj_0/
│   │   ├── mesh.glb
│   │   ├── pose.json
│   │   ├── bounds.json
│   │   └── material.json
│   └── ...
├── background/
│   └── mesh.glb
├── camera/
└── depth/
```

**Outputs:**
- `assets/scene_manifest.json` - Canonical manifest
- `layout/scene_layout_scaled.json` - Layout with transforms
- `seg/inventory.json` - Semantic inventory
- `assets/obj_*/` - Organized asset directories

### 2. Scale Authority Integration

Scale is determined by:
1. User-provided anchor (highest priority)
2. Calibrated scale factor from scale-job
3. ZeroScene scale (if trusted)
4. Reference object heuristics

The authoritative scale is written to:
- `manifest.scene.meters_per_unit`
- `layout.meters_per_unit`
- USD `metersPerUnit` metadata

### 3. Articulation Wiring

After `interactive-job` generates articulated assets:
1. Assets are placed in `assets/interactive/obj_{id}/`
2. `usd-assembly-job` automatically wires them into `scene.usda`
3. Manifest is updated with articulation metadata

### 4. Isaac Lab Task Generation

New `isaac-lab-job` generates:
- `env_cfg.py` - ManagerBasedEnv configuration
- `task_{policy}.py` - Task implementation
- `train_cfg.yaml` - Training hyperparameters
- `randomizations.py` - EventManager-compatible hooks
- `reward_functions.py` - Reward modules

## Pipeline Flow Comparison

### Current Pipeline (Gemini-based)
```
image → seg-job → multiview-job → sam3d-job → hunyuan-job
                                      ↓
                              usd-assembly-job → simready-job → replicator-job
```

### ZeroScene Pipeline
```
image → ZeroScene → zeroscene-job → simready-job → usd-assembly-job
                         ↓                              ↓
                  interactive-job                 replicator-job
                                                       ↓
                                               isaac-lab-job
```

## Migration Steps

1. **Keep Both Pipelines**: Maintain the Gemini pipeline as fallback
2. **Implement ZeroScene Adapter**: Use `tools/zeroscene_adapter/`
3. **Update Workflows**: Add ZeroScene workflow YAML
4. **Test End-to-End**: Use `tools/qa_validation/` for validation
5. **Deprecate Old Jobs**: Once ZeroScene is stable

## Definition of Done

A scene is "done" when:

- [ ] `scene.usda` loads in Isaac Sim without errors
- [ ] Scale is correct (countertops ~0.9m, doors ~2m)
- [ ] All objects have collision proxies
- [ ] Articulated objects have controllable joints
- [ ] Physics simulation stable for 100+ steps
- [ ] Replicator scripts execute and generate frames
- [ ] Isaac Lab task imports and runs reset/step

Use `tools/qa_validation/` to verify:

```python
from tools.qa_validation import run_qa_validation

report = run_qa_validation(
    scene_dir=Path("/mnt/gcs/scenes/scene_123"),
    output_report=Path("validation_report.json")
)

if report.passed:
    print("Scene validated!")
else:
    for issue in report.issues:
        print(f"FAIL: {issue}")
```

## Environment Variables

Jobs should use these environment variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `BUCKET` | GCS bucket name | `blueprint-scenes` |
| `SCENE_ID` | Scene identifier | `scene_123` |
| `ASSETS_PREFIX` | Assets path | `scenes/scene_123/assets` |
| `LAYOUT_PREFIX` | Layout path | `scenes/scene_123/layout` |
| `USD_PREFIX` | USD output path | `scenes/scene_123/usd` |
| `REPLICATOR_PREFIX` | Replicator path | `scenes/scene_123/replicator` |
| `ZEROSCENE_PREFIX` | ZeroScene output path | `scenes/scene_123/zeroscene` |
| `ENVIRONMENT_TYPE` | Environment hint | `kitchen`, `office`, etc. |
| `SCALE_FACTOR` | Scale override | `1.0` |
| `TRUST_ZEROSCENE_SCALE` | Trust ZeroScene scale | `true`/`false` |
