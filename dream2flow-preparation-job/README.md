# Dream2Flow Preparation Job

Generates Dream2Flow bundles from BlueprintPipeline scenes based on [Dream2Flow (arXiv:2512.24766)](https://arxiv.org/abs/2512.24766).

## Overview

Dream2Flow bridges the gap between high-level language instructions and low-level robot control by:

1. **Video Generation**: Using video diffusion models to "imagine" what task execution looks like
2. **3D Object Flow Extraction**: Extracting object motion from generated videos using:
   - Object segmentation (SAM, Grounded-SAM)
   - Depth estimation (DepthAnything, ZoeDepth)
   - Point tracking (CoTracker, TAPIR)
3. **Robot Control**: Using 3D object flow as an embodiment-agnostic goal/reward for:
   - Trajectory optimization (MPC, iLQR)
   - Reinforcement learning

## Key Concept: 3D Object Flow

Imagine painting tiny dots on an object. The **3D object flow** is the trajectory of each dot over time as the task is performed. This representation is:

- **Object-centric**: Describes what happens to the object, not the robot
- **Embodiment-agnostic**: Same flow can be achieved by different robots
- **Language-grounded**: Derived from natural language instructions

## Pipeline

```
RGB-D Observation + Language Instruction
              ↓
      Video Generation (Dream)
              ↓
    Object Segmentation (Masks)
              ↓
      Depth Estimation
              ↓
   Point Tracking (2D Tracks)
              ↓
    3D Lifting (Object Flow)
              ↓
Robot Tracking (Trajectory/RL)
```

## Usage

### Command Line

```bash
# Generate from scene directory
python prepare_dream2flow_bundle.py \
    --scene-dir ./scenes/kitchen_001 \
    --num-tasks 5

# Generate from manifest
python prepare_dream2flow_bundle.py \
    --manifest-path ./assets/scene_manifest.json \
    --output-dir ./dream2flow_output \
    --tasks open close push grasp

# Specify robot embodiment
python prepare_dream2flow_bundle.py \
    --scene-dir ./scenes/kitchen_001 \
    --robot franka_panda \
    --num-tasks 10
```

### Python API

```python
from prepare_dream2flow_bundle import (
    Dream2FlowJobConfig,
    Dream2FlowPreparationJob,
    prepare_dream2flow_bundles,
)

# Quick start
output = prepare_dream2flow_bundles(
    manifest_path="./assets/scene_manifest.json",
    output_dir="./dream2flow_output",
    num_tasks=5,
)

# Full configuration
config = Dream2FlowJobConfig(
    manifest_path=Path("./assets/scene_manifest.json"),
    output_dir=Path("./dream2flow_output"),
    num_tasks=10,
    task_types=[TaskType.OPEN, TaskType.GRASP, TaskType.PUSH],
    resolution=(720, 480),
    num_frames=49,
    fps=24.0,
    robot_embodiment=RobotEmbodiment.FRANKA_PANDA,
)

job = Dream2FlowPreparationJob(config)
output = job.run()

for bundle in output.bundles:
    print(f"Task: {bundle.instruction.text}")
    print(f"  Video: {bundle.video_generation_success}")
    print(f"  Flow: {bundle.flow_extraction_success}")
```

## Output Structure

```
dream2flow/
├── dream2flow_bundles_manifest.json
├── {scene_id}_task_000/
│   ├── manifest.json
│   ├── observation/
│   │   ├── initial_rgb.png
│   │   └── initial_depth.png
│   ├── video/
│   │   ├── {video_id}.mp4
│   │   └── {video_id}_frames/
│   ├── flow/
│   │   ├── masks/
│   │   ├── depth/
│   │   ├── tracks/
│   │   └── visualization/
│   └── trajectory/
│       └── {trajectory_id}.json
└── {scene_id}_task_001/
    └── ...
```

## Supported Tasks

From the Dream2Flow paper examples:
- Push-T, Sweep, Push
- Put in bowl, Cover bowl
- Open/Close (oven, drawer, door)
- Pull chair
- Recycle can
- Pick and place

## Supported Robots

- Franka Panda (default)
- UR5e, UR10
- Boston Dynamics Spot
- Fourier GR1
- Custom

## Status

**Note**: The Dream2Flow model is not yet publicly released. This module provides:

- ✅ Complete pipeline scaffolding
- ✅ Data models and bundle structure
- ✅ Placeholder video/flow generation for testing
- ✅ Cloud Run job and workflow integration
- ⏳ Full model integration (pending release)

When the Dream2Flow model is released, update:
1. `video_generator/video_generator.py` - Add model loading
2. `flow_extractor/flow_extractor.py` - Integrate vision foundation models
3. `robot_tracker/robot_tracker.py` - Add Isaac Lab integration

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| BUCKET | Yes | - | GCS bucket name |
| SCENE_ID | Yes | - | Scene identifier |
| ASSETS_PREFIX | No | scenes/{SCENE_ID}/assets | GCS prefix for assets |
| USD_PREFIX | No | scenes/{SCENE_ID}/usd | GCS prefix for USD |
| DREAM2FLOW_PREFIX | No | scenes/{SCENE_ID}/dream2flow | Output prefix |
| NUM_TASKS | No | 5 | Number of tasks to generate |
| RESOLUTION_WIDTH | No | 720 | Video width |
| RESOLUTION_HEIGHT | No | 480 | Video height |
| NUM_FRAMES | No | 49 | Frames per video |
| FPS | No | 24 | Frames per second |
| ROBOT | No | franka_panda | Robot embodiment |
| VIDEO_API_ENDPOINT | No | - | Remote video generation API |

## References

- [Dream2Flow Paper (arXiv:2512.24766)](https://arxiv.org/abs/2512.24766)
- [DWM Paper (arXiv:2512.17907)](https://arxiv.org/abs/2512.17907) - Related work
- [BlueprintPipeline Documentation](../README.md)
