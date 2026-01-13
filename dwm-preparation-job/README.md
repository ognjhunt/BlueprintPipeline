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

## Dependencies & Availability

- **Isaac Sim rendering is required for production bundles.** Static scene videos must be rendered against the real USD scene using the Isaac Sim backend. The mock renderer is intended only for CI/smoke tests where GPU rendering is unavailable.
- **DWM model availability gates inference.** The repository prepares conditioning bundles today, but running DWM inference requires access to the released model weights/API. Until the model is available, inference should be treated as optional or disabled.

### Local Pipeline
```bash
python tools/run_local_pipeline.py --scene-dir ./scene --steps dwm
```

### Standalone CLI
```bash
python dwm-preparation-job/prepare_dwm_bundle.py \
    --scene-dir ./scenes/kitchen_001 \
    --num-trajectories 10
```

### Cloud Auto-Trigger (Production)

The job auto-triggers when 3D-RE-GEN completes via EventArc:

```
.regen3d_complete marker uploaded
        ↓ (EventArc trigger)
dwm-preparation-pipeline workflow
        ↓
dwm-preparation-job Cloud Run job
        ↓
.dwm_complete marker written
```

**Deploy to Cloud Run:**
```bash
gcloud builds submit --config=dwm-preparation-job/cloudbuild.yaml .
```

**Setup EventArc Trigger:**
```bash
chmod +x dwm-preparation-job/scripts/setup_eventarc_trigger.sh
./dwm-preparation-job/scripts/setup_eventarc_trigger.sh <project_id> <bucket_name>
```

Or manually:
```bash
gcloud eventarc triggers create dwm-preparation-trigger \
  --location=us-central1 \
  --service-account="${WORKFLOW_SA}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --destination-workflow=dwm-preparation-pipeline \
  --destination-workflow-location=us-central1 \
  --event-filters="type=google.cloud.storage.object.v1.finalized" \
  --event-filters="bucket=${BUCKET}" \
  --event-data-content-type="application/json"
```

**Environment Variables (Cloud Run):**

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| BUCKET | Yes | - | GCS bucket name |
| SCENE_ID | Yes | - | Scene identifier |
| ASSETS_PREFIX | No | scenes/{id}/assets | GCS prefix for assets |
| USD_PREFIX | No | scenes/{id}/usd | GCS prefix for USD files |
| DWM_PREFIX | No | scenes/{id}/dwm | GCS prefix for output |
| NUM_TRAJECTORIES | No | 5 | Trajectories per scene |
| RESOLUTION_WIDTH | No | 720 | Video width |
| RESOLUTION_HEIGHT | No | 480 | Video height |
| NUM_FRAMES | No | 49 | Frames per video |
| FPS | No | 24 | Frames per second |
| MANO_MODEL_PATH | No | /mano_models | Path to MANO files |
| SKIP_DWM | No | false | Skip DWM preparation entirely |

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

### MANO Hand Model (Optional, Recommended)

For anatomically accurate hand meshes (instead of geometric placeholders), you need the MANO model files.

**Quick Setup:**
1. Register at [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de/)
2. Download MANO model files
3. Install: `pip install smplx torch`
4. Place files in `~/.mano/models/` or set `MANO_MODEL_PATH`

**Detailed Instructions:** See [MANO_SETUP.md](MANO_SETUP.md) for:
- Cloud Run deployment with MANO
- GCS-based model file loading
- License considerations
- Troubleshooting guide

Without MANO files, the renderer falls back to SimpleHandMesh (geometric boxes) automatically.

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
- MANO integration is ready (stub implemented), but requires:
  - `smplx` package installation
  - MANO model files from MPI (license required)
  - Falls back to SimpleHandMesh (geometric boxes) if MANO unavailable
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
