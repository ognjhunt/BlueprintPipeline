# Running SceneSmith on RunPod (Stage 1 Only)

Step-by-step guide to run the SceneSmith text-to-scene pipeline on a RunPod pod with SAM3D assets and OpenRouter/Minimax LLM.

## Prerequisites

- RunPod pod with **L40S GPU** (48GB VRAM) — `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` image
- API keys:
  - `OPENROUTER_API_KEY` — routes to Minimax M2.5 via OpenRouter
  - `GOOGLE_API_KEY` — for Gemini context image generation
  - `HF_TOKEN` (optional) — for downloading SAM3D checkpoints from HuggingFace
  - `GITHUB_TOKEN` (optional) — for cloning private BlueprintPipeline repo

## Quick Start (5 commands)

If the pod already has a completed bootstrap (`.bootstrap_complete` marker exists):

```bash
# 1. Source environment
source /workspace/.env
cd /workspace/scenesmith && source .venv/bin/activate

# 2. Kill any zombie processes from previous runs
fuser -k 7005/tcp 7006/tcp 7007/tcp 7008/tcp 7009/tcp 2>/dev/null || true

# 3. Run SceneSmith with quality overrides
bash /workspace/BlueprintPipeline/scripts/run-scenesmith-quality.sh \
  --prompt "A modern kitchen with marble countertops, stainless steel appliances, and a kitchen island with bar stools" \
  --name kitchen_test

# Or run directly:
python main.py \
  +name=my_scene \
  'experiment.prompts=["A cozy living room with a sectional sofa and fireplace"]' \
  experiment.num_workers=1 \
  experiment.pipeline.parallel_rooms=false \
  furniture_agent.asset_manager.general_asset_source=generated \
  furniture_agent.asset_manager.backend=sam3d \
  furniture_agent.asset_manager.router.strategies.generated.enabled=true \
  furniture_agent.asset_manager.router.strategies.articulated.enabled=false \
  furniture_agent.asset_manager.image_generation.backend=gemini \
  furniture_agent.context_image_generation.enabled=true \
  wall_agent.asset_manager.general_asset_source=generated \
  wall_agent.asset_manager.backend=sam3d \
  wall_agent.asset_manager.router.strategies.generated.enabled=true \
  wall_agent.asset_manager.router.strategies.articulated.enabled=false \
  wall_agent.asset_manager.image_generation.backend=gemini \
  ceiling_agent.asset_manager.general_asset_source=generated \
  ceiling_agent.asset_manager.backend=sam3d \
  ceiling_agent.asset_manager.router.strategies.generated.enabled=true \
  ceiling_agent.asset_manager.router.strategies.articulated.enabled=false \
  ceiling_agent.asset_manager.image_generation.backend=gemini \
  manipuland_agent.asset_manager.general_asset_source=generated \
  manipuland_agent.asset_manager.backend=sam3d \
  manipuland_agent.asset_manager.router.strategies.generated.enabled=true \
  manipuland_agent.asset_manager.router.strategies.articulated.enabled=false \
  manipuland_agent.asset_manager.image_generation.backend=gemini
```

## Full Setup (Fresh Pod)

### Step 1: Write API Keys

```bash
cat > /workspace/.env << 'EOF'
export LIDRA_SKIP_INIT=1
export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1
export OPENROUTER_API_KEY="<your-openrouter-key>"
export GOOGLE_API_KEY="<your-google-api-key>"
export HF_TOKEN="<your-huggingface-token>"
export GITHUB_TOKEN="<your-github-pat>"
EOF
chmod 600 /workspace/.env
source /workspace/.env
```

### Step 2: Install System Libraries

```bash
apt-get update -qq
apt-get install -y -qq \
    libpython3.11-dev libxrender1 libxi6 libxxf86vm1 libxfixes3 libgl1 \
    libxkbcommon0 libsm6 libice6 libxext6 libxrandr2 libxcursor1 \
    libxinerama1 libepoxy0 libglu1-mesa libegl1 libegl-mesa0 \
    libgles2-mesa libopengl0 libglx-mesa0 psmisc git-lfs
apt-get remove -y bubblewrap 2>/dev/null || true
```

### Step 3: Install uv Package Manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Cache on persistent volume
rm -rf /root/.cache/uv
mkdir -p /workspace/.cache/uv
ln -s /workspace/.cache/uv /root/.cache/uv
export UV_LINK_MODE=copy
```

### Step 4: Clone SceneSmith

```bash
git clone --depth 1 https://github.com/nepfaff/scenesmith.git /workspace/scenesmith
cd /workspace/scenesmith
git submodule update --init --recursive
```

### Step 5: Install Python Dependencies

```bash
cd /workspace/scenesmith
uv sync --no-dev
source .venv/bin/activate
```

### Step 6: Build GPU Packages (~15-20 min)

```bash
# gsplat
uv pip install --no-build-isolation \
    "git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7"

# nvdiffrast
uv pip install --no-build-isolation \
    "git+https://github.com/NVlabs/nvdiffrast.git"

# pytorch3d (slowest - ~10 min)
FORCE_CUDA=1 uv pip install --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git"

# kaolin
uv pip install kaolin \
    -f "https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html"
```

### Step 7: Pin xformers + torch

```bash
uv pip install 'xformers==0.0.28.post3' \
    --index-url https://download.pytorch.org/whl/cu124
uv pip install 'torch==2.5.1+cu124' 'torchvision==0.20.1+cu124' \
    --index-url https://download.pytorch.org/whl/cu124
```

### Step 8: Install SAM3D Dependencies

```bash
uv pip install -e external/sam-3d-objects/ --no-deps
uv pip install \
    open3d optree roma loguru astor einops-exts point-cloud-utils \
    scikit-image trimesh easydict einops fvcore plyfile spconv-cu120 \
    timm lightning pyvista pymeshfix igraph
uv pip install \
    'MoGe @ git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b'
```

### Step 9: Download Checkpoints

```bash
cd /workspace/scenesmith
mkdir -p external/checkpoints

# SAM3D model checkpoint
huggingface-cli download facebook/sam3 sam3.pt \
    --local-dir external/checkpoints --token "${HF_TOKEN}"

# SAM-3D-Objects checkpoints
huggingface-cli download facebook/sam-3d-objects \
    --repo-type model \
    --local-dir external/checkpoints/sam-3d-objects-dl \
    --include 'checkpoints/*' --token "${HF_TOKEN}"
cp external/checkpoints/sam-3d-objects-dl/checkpoints/* external/checkpoints/
rm -rf external/checkpoints/sam-3d-objects-dl

# AmbientCG materials
python3 scripts/download_ambientcg.py --output data/materials -r 1K -f JPG -c 8
```

### Step 10: (Optional) Clone BlueprintPipeline

Only needed if you want to use the wrapper scripts:

```bash
git clone --depth 1 https://${GITHUB_TOKEN}@github.com/ognjhunt/BlueprintPipeline.git /workspace/BlueprintPipeline
```

### Step 11: Mark Bootstrap Complete

```bash
touch /workspace/.bootstrap_complete
```

## Running SceneSmith

### Environment Variables That Control Behavior

| Variable | Value | Purpose |
|----------|-------|---------|
| `OPENAI_API_KEY` | auto from `OPENROUTER_API_KEY` | Passed to agents |
| `OPENAI_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter endpoint |
| `GOOGLE_API_KEY` | your key | Gemini context images |

### Hydra Override Reference (Quality Mode)

For each agent prefix (`furniture_agent`, `wall_agent`, `ceiling_agent`, `manipuland_agent`):

```
{prefix}.asset_manager.general_asset_source=generated    # Force generation (no retrieval)
{prefix}.asset_manager.backend=sam3d                     # Use SAM3D for 3D mesh generation
{prefix}.asset_manager.router.strategies.generated.enabled=true
{prefix}.asset_manager.router.strategies.articulated.enabled=false
{prefix}.asset_manager.articulated.sources.partnet_mobility.enabled=false
{prefix}.asset_manager.articulated.sources.artvip.enabled=false
{prefix}.asset_manager.image_generation.backend=gemini   # Gemini for reference images
```

Plus for furniture agent:
```
furniture_agent.context_image_generation.enabled=true     # Room-level Gemini guidance
```

### LLM Model Override

To change the LLM model, set before running:
```bash
export OPENAI_API_KEY="${OPENROUTER_API_KEY}"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
```

Then pass model override:
```
+model=minimax/minimax-m2.5
```

Or SceneSmith reads `OPENAI_API_KEY` + `OPENAI_BASE_URL` directly.

### Output Location

Results are saved in the SceneSmith `outputs/` directory:
```
/workspace/scenesmith/outputs/<name>/
├── house_state.json        # Object list with poses
├── *.glb                   # Generated 3D meshes (SAM3D)
├── rendered_views/         # Rendered images
└── logs/                   # Agent conversation logs
```

## Downloading Results to Mac

From your Mac:
```bash
# Using the RunPod proxy SSH (SCP not supported - use tar + base64)
# First, on the pod:
cd /workspace/scenesmith/outputs && tar czf /tmp/scene_output.tar.gz <scene_name>/

# Then download via SSH piping:
ssh -o RequestTTY=force tyb4t32f4tn2t8-64411879@ssh.runpod.io \
    -i ~/.ssh/id_ed25519 \
    "cat /tmp/scene_output.tar.gz" > ~/Downloads/scene_output.tar.gz

# Or if direct TCP SSH works:
scp -P 50571 root@160.250.71.210:/tmp/scene_output.tar.gz ~/Downloads/
```

## Monitoring

```bash
# Watch the bootstrap log
tail -f /tmp/bootstrap.log

# Watch the SceneSmith run
# (SceneSmith logs to stdout, or redirect to a file)
```

## Troubleshooting

### "Permission denied" on SSH
The RunPod proxy SSH requires `-o RequestTTY=force`. Direct TCP SSH (port 50571) needs the correct SSH key registered in RunPod account settings.

### SAM3D checkpoints missing
Set `HF_TOKEN` and re-run Step 9. Check `ls /workspace/scenesmith/external/checkpoints/`.

### GPU out of memory
SAM3D + Gemini + L40S 48GB should be fine. If OOM occurs, reduce `experiment.num_workers=1` and ensure no other GPU processes are running.

### OpenRouter rate limits
Minimax M2.5 via OpenRouter has generous limits. If rate limited, add delays or switch model:
```
+model=openai/gpt-4o-mini
```

## RunPod Pod Details

- **Pod ID**: `tyb4t32f4tn2t8`
- **Name**: `blueprint-l40s-migration`
- **Image**: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- **GPU**: NVIDIA L40S (48GB)
- **SSH Proxy**: `tyb4t32f4tn2t8-64411879@ssh.runpod.io`
- **Direct TCP**: `root@160.250.71.210:50571`
- **Web Terminal**: Port 19123

## Automated Bootstrap Script

For fully automated setup, use:
```bash
bash /workspace/BlueprintPipeline/scripts/runpod-bootstrap-stage1.sh
```

This handles all steps above and writes a `.bootstrap_complete` marker when done.
On subsequent pod restarts, it detects the marker and only reinstalls system libs (lost on container restart).
