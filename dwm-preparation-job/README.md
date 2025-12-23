# DWM Preparation Job

Generates **DWM (Dexterous World Model)** conditioning data from BlueprintPipeline scenes.

## What is DWM?

DWM ([arXiv:2512.17907](https://arxiv.org/abs/2512.17907)) is a video diffusion model that generates egocentric interaction videos. Given:
1. **Static scene video** - rendered from a 3D scene along a camera trajectory
2. **Hand mesh video** - rendered hand meshes along the same trajectory
3. **Text prompt** - semantic description of the action

DWM generates realistic videos showing plausible hand-scene interaction dynamics (e.g., opening drawers, grasping objects, pushing buttons).

## Why This Matters for Robotics

- **Sim2Real Gap**: DWM offers a different approach to simulation - instead of physics engines with friction/contact bugs, it learns visual dynamics from data
- **Egocentric Data**: Robotics labs are seeking egocentric video data, which is expensive to capture in the real world
- **Digital Twin Activation**: Turn static 3D reconstructions into "interactive" experiences without explicit articulation modeling

## Pipeline Integration

This job runs as part of the BlueprintPipeline:

```
Image → 3D-RE-GEN → BlueprintPipeline → ... → DWM Preparation → DWM Bundles
```

Run with the local pipeline:
```bash
python tools/run_local_pipeline.py --scene-dir ./scene --steps dwm
```

Or standalone:
```bash
python dwm-preparation-job/prepare_dwm_bundle.py \
    --scene-dir ./scenes/kitchen_001 \
    --num-trajectories 10
```

## Output Structure

For each scene, generates multiple DWM conditioning bundles:

```
dwm/
├── dwm_bundles_manifest.json          # Overall manifest
├── {scene_id}_{trajectory_id}/
│   ├── manifest.json                  # Bundle metadata
│   ├── static_scene_video.mp4         # Rendered static scene
│   ├── hand_mesh_video.mp4            # Rendered hand meshes
│   ├── camera_trajectory.json         # Camera poses (per frame)
│   ├── hand_trajectory.json           # Hand poses (MANO-compatible)
│   ├── frames/
│   │   ├── static_scene/              # Individual PNG frames
│   │   └── hand_mesh/                 # Individual PNG frames
│   └── metadata/
│       ├── prompt.txt                 # Text prompt for DWM
│       └── bundle_info.json           # Additional metadata
```

## Key Components

### 1. Trajectory Generation (`trajectory_generator/`)
Generates egocentric camera trajectories:
- **Approach**: Walk toward an object
- **Orbit**: Circle around an object
- **Reach-Manipulate**: Approach + interaction phases
- **Walkthrough**: Linear path through scene

### 2. Scene Rendering (`scene_renderer/`)
Renders static 3D scene along camera trajectories:
- Supports Isaac Sim, PyRender, or mock rendering
- Outputs 720×480 frames at 24fps (DWM default)
- 49 frames per trajectory (DWM default)

### 3. Hand Motion (`hand_motion/`)
Generates and renders hand manipulation trajectories:
- **Actions**: Reach, Grasp, Pull, Push, Rotate, Lift, Place
- MANO-compatible pose parameters
- Aligned with camera trajectories

### 4. Bundle Packaging (`bundle_packager/`)
Packages all components into DWM-ready bundles:
- Generates text prompts based on action type
- Creates manifests for downstream processing
- Optional video encoding and compression

## Configuration

Default parameters (matching DWM paper):
- **Resolution**: 720×480
- **Frames**: 49
- **FPS**: 24
- **Trajectories per scene**: 5

Override via CLI:
```bash
python prepare_dwm_bundle.py \
    --manifest-path ./assets/scene_manifest.json \
    --resolution 1280 720 \
    --num-frames 49 \
    --fps 24 \
    --num-trajectories 10 \
    --actions grasp pull push
```

## Dependencies

Required:
- numpy
- PIL/Pillow (for rendering)

Optional:
- pyrender (better rendering quality)
- ffmpeg (video encoding)
- trimesh (mesh loading)

## DWM Integration (When Code is Released)

Once DWM code is available, bundles can be used directly:

```python
# Load bundle
bundle_path = Path("dwm/scene_traj_001/")
manifest = json.loads((bundle_path / "manifest.json").read_text())

# Get conditioning inputs
static_video = bundle_path / manifest["static_scene_video"]
hand_video = bundle_path / manifest["hand_mesh_video"]
prompt = (bundle_path / "metadata/prompt.txt").read_text()

# Run DWM inference (pseudocode - actual API TBD)
output_video = dwm.generate(
    static_scene_video=static_video,
    hand_mesh_video=hand_video,
    text_prompt=prompt,
)
```

## Limitations

Current implementation:
- Uses simple geometric hand model (MANO integration planned)
- Mock renderer used when PyRender/Isaac Sim unavailable
- USD scene rendering requires Isaac Sim

DWM itself (from paper):
- Output is video only, not updated 3D state
- No explicit physics/contact reasoning
- May struggle with deformable objects

## References

- **DWM Paper**: [arXiv:2512.17907](https://arxiv.org/abs/2512.17907)
- **DWM Project**: [snuvclab.github.io/dwm](https://snuvclab.github.io/dwm/)
- **3D-RE-GEN**: [3dregen.jdihlmann.com](https://3dregen.jdihlmann.com/)
