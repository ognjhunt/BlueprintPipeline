# Texture Generation Investigation (2026-02-10)

## Scope
Investigate why `enable_texture_hy21` remains disabled and determine patch-ready remediation for re-enabling texture output safely.

## Observations Collected

### 1) Texture currently disabled by default
- `configs/regen3d_reconstruct.env`:
  - `REGEN3D_ENABLE_TEXTURE=false`
  - comment notes: "Hunyuan3D UNet type mismatch causes pipeline hang"
- `tools/regen3d_runner/runner.py` default:
  - `enable_texture_hy21: bool = False`

### 2) VM runtime dependency versions (captured live)
Command:
```bash
gcloud compute ssh isaac-sim-ubuntu --zone=us-east1-c --command \
  "/home/nijelhunt1/3D-RE-GEN/venv_py310/bin/pip show torch diffusers transformers accelerate xformers 2>/dev/null | egrep '^(Name|Version):'"
```

Result:
- `torch==2.6.0+cu124`
- `diffusers==0.36.0`
- `transformers==5.1.0`
- `accelerate==1.12.0`
- `xformers` not reported by `pip show` (not installed in this venv)

### 3) Upstream dependency constraints in VM repo are weak/unpinned for texture stack
Command:
```bash
gcloud compute ssh isaac-sim-ubuntu --zone=us-east1-c --command \
  "grep -nEi 'diffusers|torch|transformers|xformers|hunyuan' /home/nijelhunt1/3D-RE-GEN/requirements.txt /home/nijelhunt1/3D-RE-GEN/segmentor/requirements.txt"
```

Result highlights:
- `segmentor/requirements.txt` includes `diffusers`, `torch`, `torchvision`, `transformers` with no version pins.
- `requirements.txt` comments reference torch install strategy, but no strict lock for this texture path.

### 4) Prior logs did not preserve a texture traceback
Searches in retained `/tmp/*.log` and current laundry output tree found no persisted UNet/diffusers stack trace.

## Additional blocker discovered during this session
- Remote setup for SAM3/prewarm repeatedly hit a 1200s timeout while PyTorch3D was still compiling.
- This was fixed in local runner/config by adding and wiring:
  - `REGEN3D_SETUP_TIMEOUT_S` (default `3600`)
- While not the root cause of texture failure, this was preventing reliable repro runs.

## Repro Plan (single-scene, minimal)
After SAM3/setup completion, run a minimal texture-enabled reconstruction on Laundry image with reduced labels/steps to force the Hunyuan texture path and capture full traceback.

Planned local invocation:
```bash
REGEN3D_AUTO_START_VM=false \
REGEN3D_ENABLE_TEXTURE=true \
REGEN3D_STEPS=1,2,4 \
REGEN3D_LABELS=detergent_bottle \
python - <<'PY'
from pathlib import Path
from tools.regen3d_runner.runner import Regen3DRunner

scene = "texture_repro_20260210"
runner = Regen3DRunner()
res = runner.run_reconstruction(
    input_image=Path('/tmp/blueprint_scenes/ChIJHy53k-XlrIkRTdgT1Ev8ln4/input/room.png'),
    scene_id=scene,
    output_dir=Path('/tmp/regen3d_texture_repro_20260210'),
)
print('success=', res.success)
print('error=', res.error)
print('remote_log_tail=', res.remote_log[-4000:])
PY
```

## Patch-Ready Direction (pending final traceback)
If traceback confirms diffusers/Hunyuan incompatibility:
1. Pin known-good texture stack in `tools/regen3d_runner/setup_remote.sh` (torch/diffusers/transformers/accelerate and any Hunyuan-specific deps).
2. Add a fast verification command in bootstrap:
   - import texture pipeline modules
   - instantiate expected UNet/pipeline classes
   - fail setup early if class contract mismatch occurs.
3. Keep `REGEN3D_ENABLE_TEXTURE=false` as default until one full scene finishes with textured GLBs and no hangs.

## Status
- Dependency evidence captured.
- Historical traceback not available.
- Active repro execution pending completion of current VM setup bootstrap.
