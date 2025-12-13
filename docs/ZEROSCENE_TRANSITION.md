# ZeroScene Pipeline Guide

This document describes the ZeroScene-based BlueprintPipeline for scene reconstruction and SimReady asset generation.

**Last Updated**: December 2025

## Overview

[ZeroScene](https://arxiv.org/html/2509.23607v1) provides the reconstruction pipeline that covers:
- Instance segmentation + depth extraction
- Object pose optimization using 3D/2D projection losses
- Foreground + background mesh reconstruction
- PBR material estimation for improved rendering realism
- Triangle mesh outputs for each object

## Pipeline Architecture

### Job Sequence

```
image → ZeroScene → zeroscene-job → scale-job (optional) → interactive-job
                                                                    ↓
                                                             simready-job
                                                                    ↓
                                                           usd-assembly-job
                                                                    ↓
                                                            replicator-job
                                                                    ↓
                                                           variation-gen-job
                                                                    ↓
                                                            isaac-lab-job
```

### Active Jobs

| Job | Purpose | Notes |
|-----|---------|-------|
| `zeroscene-job` | Adapter for ZeroScene outputs | Converts ZeroScene → manifest + layout |
| `scale-job` | Scale calibration (optional) | Calibrates metric scale if needed |
| `interactive-job` | Articulation bridge (PhysX-Anything) | Adds joints to doors/drawers |
| `simready-job` | Physics + manipulation hints | Adds mass/friction/graspability |
| `usd-assembly-job` | Convert + assemble scene.usda | Final USD assembly |
| `replicator-job` | Policy scripts + placement regions | Domain randomization |
| `variation-gen-job` | Domain randomization assets | Generates clutter objects |
| `isaac-lab-job` | Isaac Lab task generation | RL training packages |

## Tooling

### LLM Client (`tools/llm_client/`)

Unified API for Gemini + OpenAI GPT-5.2 Thinking:

| Provider | Model | Use Case |
|----------|-------|----------|
| **Google Gemini** | gemini-3-pro-preview | Default for all jobs |
| **OpenAI** | gpt-5.2-thinking | Alternative with web browsing |

**Environment Variables:**
```bash
# Select LLM provider
LLM_PROVIDER=auto          # Auto-detect (default)
LLM_PROVIDER=gemini        # Force Gemini
LLM_PROVIDER=openai        # Force OpenAI GPT-5.2 Thinking

# API Keys (at least one required)
GEMINI_API_KEY=your_key
OPENAI_API_KEY=your_key

# Fallback behavior
LLM_FALLBACK_ENABLED=true  # Auto-fallback to other provider
```

**Usage in Jobs:**
```python
from tools.llm_client import create_llm_client

# Auto-select provider
client = create_llm_client()

# Generate content with web search
response = client.generate(
    prompt="Analyze this scene",
    image=pil_image,
    json_output=True,
    use_web_search=True  # Enables Google Search or web browsing
)
```

### Job Registry (`tools/job_registry/`)

Central registry tracking all pipeline jobs:

```python
from tools.job_registry import get_registry

registry = get_registry()

# Get all active jobs
jobs = registry.get_active_jobs()

# Check pipeline readiness
if registry.is_zeroscene_ready():
    print("ZeroScene pipeline is ready")

# Get job sequence
sequence = registry.get_job_sequence()

# Print status report
registry.print_status_report()
```

### Pipeline Selector (`tools/pipeline_selector/`)

Handles job routing:

```python
from tools.pipeline_selector import select_pipeline
from pathlib import Path

decision = select_pipeline(Path('/mnt/gcs/scenes/scene_123'))
print(f'Use ZeroScene: {decision.use_zeroscene}')
print(f'Jobs: {decision.job_sequence}')
```

## ZeroScene Adapter Job

The adapter converts ZeroScene outputs to BlueprintPipeline format:

### Input Structure
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

### Output Structure
- `assets/scene_manifest.json` - Canonical manifest
- `layout/scene_layout_scaled.json` - Layout with transforms
- `seg/inventory.json` - Semantic inventory
- `assets/obj_*/` - Organized asset directories

## Scale Authority

Scale is determined by priority:
1. User-provided anchor (highest priority)
2. Calibrated scale factor from scale-job
3. ZeroScene scale (if trusted via `TRUST_ZEROSCENE_SCALE=true`)
4. Reference object heuristics

The authoritative scale is written to:
- `manifest.scene.meters_per_unit`
- `layout.meters_per_unit`
- USD `metersPerUnit` metadata

## Articulation Wiring

After `interactive-job` generates articulated assets:
1. Assets are placed in `assets/interactive/obj_{id}/`
2. `usd-assembly-job` automatically wires them into `scene.usda`
3. Manifest is updated with articulation metadata

## Isaac Lab Task Generation

`isaac-lab-job` generates:
- `env_cfg.py` - ManagerBasedEnv configuration
- `task_{policy}.py` - Task implementation
- `train_cfg.yaml` - Training hyperparameters
- `randomizations.py` - EventManager-compatible hooks
- `reward_functions.py` - Reward modules

## Environment Variables

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
| `LLM_PROVIDER` | LLM provider selection | `auto`, `gemini`, `openai` |
| `ZEROSCENE_AVAILABLE` | Override ZeroScene detection | `true`/`false` |

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

## CLI Usage

Check pipeline status from command line:

```bash
# Print job registry status
python -c "from tools.job_registry import get_registry; get_registry().print_status_report()"

# Select pipeline for a scene
python -c "
from tools.pipeline_selector import select_pipeline
from pathlib import Path

decision = select_pipeline(Path('/mnt/gcs/scenes/scene_123'))
print(f'Use ZeroScene: {decision.use_zeroscene}')
print(f'Jobs: {decision.job_sequence}')
"
```
