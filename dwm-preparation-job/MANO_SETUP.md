# MANO Hand Model Setup

This guide explains how to set up the MANO hand model for anatomically accurate hand rendering in the DWM preparation pipeline.

## What is MANO?

**MANO** (Model of the Articulated Hand) is a learned parametric model of hand shape and pose:
- Developed at MPI Perceiving Systems
- Published in SIGGRAPH Asia 2017
- Standard in hand-pose research (used by DWM, HaMeR, ARCTIC, etc.)
- Provides realistic 778-vertex hand mesh with 15 joints

**Reference**: [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de/)

## Why MANO for DWM?

DWM (Dexterous World Model) uses hand mesh videos as conditioning input. The quality of hand rendering affects:
- **Training quality**: Better hand meshes → better video generation
- **Action fidelity**: MANO supports anatomically accurate finger articulation
- **Consistency**: Standard format compatible with DWM paper's training data

Without MANO, the pipeline falls back to `SimpleHandMesh` (geometric box approximation), which works but produces less realistic conditioning data.

## Setup Options

### Option 1: Local Development

1. **Register for MANO access**:
   - Go to [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de/)
   - Click "Register" and create an account
   - Accept the license agreement (research use only)
   - Download the MANO model files

2. **Install smplx package**:
   ```bash
   pip install smplx torch
   ```

3. **Place model files**:
   ```bash
   # Create models directory
   mkdir -p ~/.mano/models

   # Extract downloaded zip to this location
   # Your structure should look like:
   # ~/.mano/models/
   # ├── MANO_RIGHT.pkl
   # └── MANO_LEFT.pkl
   ```

4. **Verify installation**:
   ```python
   from hand_motion.hand_mesh_renderer import HandMeshRenderer, HandRenderConfig, HandModel

   config = HandRenderConfig(hand_model=HandModel.MANO)
   renderer = HandMeshRenderer(config=config)

   # If no error, MANO is working
   print("MANO loaded successfully!")
   ```

### Option 2: Cloud Run Deployment

For production deployment, MANO files should be loaded at runtime from GCS or Secret Manager.

#### Method A: GCS Bucket (Recommended)

1. **Upload MANO files to GCS**:
   ```bash
   # Create a private bucket for model files
   gsutil mb -l us-central1 gs://${PROJECT_ID}-model-files

   # Upload MANO files
   gsutil cp ~/.mano/models/MANO_RIGHT.pkl gs://${PROJECT_ID}-model-files/mano/
   gsutil cp ~/.mano/models/MANO_LEFT.pkl gs://${PROJECT_ID}-model-files/mano/

   # Restrict access (models are licensed)
   gsutil iam ch serviceAccount:${CLOUD_RUN_SA}:objectViewer gs://${PROJECT_ID}-model-files
   ```

2. **Update entrypoint to download at startup**:
   ```python
   # In entrypoint.py, add before running job:
   def download_mano_models():
       from google.cloud import storage
       client = storage.Client()
       bucket = client.bucket(f"{os.environ['GOOGLE_CLOUD_PROJECT']}-model-files")

       mano_dir = Path("/mano_models")
       mano_dir.mkdir(exist_ok=True)

       for model_file in ["MANO_RIGHT.pkl", "MANO_LEFT.pkl"]:
           blob = bucket.blob(f"mano/{model_file}")
           if blob.exists():
               blob.download_to_filename(str(mano_dir / model_file))
               print(f"Downloaded {model_file}")

       return mano_dir
   ```

3. **Set environment variable**:
   ```yaml
   # In cloudbuild.yaml, add to gcloud run jobs create/update:
   --set-env-vars=MANO_MODEL_PATH=/mano_models
   ```

#### Method B: Bake into Docker Image (Private Registry Only)

**WARNING**: MANO license prohibits redistribution. Only use this method with a private container registry.

1. **Update Dockerfile**:
   ```dockerfile
   # Add MANO files (must be private registry!)
   COPY mano_models/ /mano_models/
   ENV MANO_MODEL_PATH=/mano_models
   ```

2. **Ensure private registry**:
   ```bash
   # Artifact Registry should be private by default
   gcloud artifacts repositories describe blueprint-jobs \
     --location=us-central1 \
     --format='value(name)'
   ```

### Option 3: Skip MANO (Fallback)

If MANO files are unavailable, the pipeline automatically falls back to `SimpleHandMesh`:

```python
# Automatic fallback in hand_mesh_renderer.py:
if not mano_available:
    print("MANO not available - using SimpleHandMesh")
    self.mesh_generator = SimpleHandMesh()
```

The fallback uses geometric approximations (palm box + finger cylinders), which is sufficient for:
- Development and testing
- Scenes where hand appearance is less critical
- Quick iteration without license setup

## File Locations

The renderer checks these locations in order:

1. `MANO_MODEL_PATH` environment variable
2. `~/.mano/models/` (Linux/Mac)
3. `./mano_models/` (current directory)
4. `/mano_models/` (Docker container)

## Troubleshooting

### "smplx not installed"
```bash
pip install smplx torch
```

### "MANO model files not found"
- Verify files exist at expected path
- Check file permissions
- Ensure both `MANO_RIGHT.pkl` and `MANO_LEFT.pkl` are present

### "License error" or "Pickle load error"
- Re-download from official MANO website
- Don't use files from unofficial sources (may be corrupted or incompatible)

### "CUDA not available"
MANO works with CPU (slower but functional):
```python
import torch
torch.set_default_device('cpu')
```

## License Considerations

**MANO License Summary** (not legal advice):
- ✅ Academic research use
- ✅ Non-commercial applications
- ❌ Commercial redistribution
- ❌ Training commercial models (requires separate license)

For commercial use, contact MPI-IS licensing:
- Email: ps-license@tue.mpg.de
- Website: https://mano.is.tue.mpg.de/license

## Integration with DWM Pipeline

Once MANO is set up, enable it in your configuration:

```python
from hand_motion.hand_mesh_renderer import HandRenderConfig, HandModel

# Explicit MANO usage
config = HandRenderConfig(
    hand_model=HandModel.MANO,
    mano_model_path="/path/to/mano/models",
)

# Or use helper method
config = HandRenderConfig().with_mano("/path/to/mano/models")
```

The pipeline will generate MANO-format hand trajectories in each bundle:
```
dwm/{bundle_id}/
├── hand_mesh_video.mp4      # Rendered MANO meshes
├── hand_trajectory.json     # MANO pose parameters
│   ├── poses[]              # Per-frame joint rotations (48 params)
│   ├── shapes[]             # Shape coefficients (10 params)
│   └── trans[]              # Global translation (3 params)
└── ...
```

These can be directly used with DWM training once the model code is released.
