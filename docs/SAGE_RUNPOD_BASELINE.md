# RunPod SAGE Baseline (TRELLIS + Isaac Sim + SAGE)

This repo includes automation to stand up a **2-pod** baseline on RunPod:

1. **Pod A (L40S)**: TRELLIS HTTP server (port `8080`)
2. **Pod B (L40S)**: Isaac Sim `nvcr.io/nvidia/isaac-sim:5.1.0` + NVlabs/SAGE client/server scripts

It then runs:
- Baseline 1: text-only room generation
- Baseline 2: image-conditioned room generation (optional, if you pass `--image`)

## Prereqs

Secrets (provide via env vars or `--secrets-env-file`):
- `HF_TOKEN` (HuggingFace token, used by TRELLIS)
- `OPENAI_API_KEY` (hosted OpenAI for agent + server calls)
- `NGC_API_KEY` (NVIDIA NGC key to pull Isaac Sim container)

RunPod:
- `RUNPOD_API_KEY` (loaded automatically from `configs/runpod_credentials.env` if present)

## Quick Start

1. Create a secrets file (gitignored; do not commit):

```bash
cat > /tmp/sage_runpod_secrets.env <<'EOF'
export HF_TOKEN="..."
export OPENAI_API_KEY="..."
export NGC_API_KEY="..."
EOF
chmod 600 /tmp/sage_runpod_secrets.env
```

2. Bootstrap pods only (recommended first):

```bash
bash /Users/nijelhunt_1/workspace/BlueprintPipeline/scripts/runpod-sage-baseline.sh \
  --secrets-env-file /tmp/sage_runpod_secrets.env \
  --skip-runs \
  --keep-pods
```

3. Run text-only baseline (will take time; uses OpenAI + TRELLIS):

```bash
bash /Users/nijelhunt_1/workspace/BlueprintPipeline/scripts/runpod-sage-baseline.sh \
  --secrets-env-file /tmp/sage_runpod_secrets.env \
  --room-desc "A medium-sized kitchen."
```

4. Run image-conditioned baseline:

```bash
bash /Users/nijelhunt_1/workspace/BlueprintPipeline/scripts/runpod-sage-baseline.sh \
  --secrets-env-file /tmp/sage_runpod_secrets.env \
  --room-desc "A medium-sized kitchen." \
  --image "/absolute/path/to/ref.png" \
  --image-room-desc "Reconstruct a semantically coherent version of this room."
```

## What Gets Produced

On Pod B:
- SAGE results: `/workspace/SAGE/server/results/<layout_id>/`
- SAGE client logs: `/workspace/SAGE/client/logs/`
- SAGE room-desc records: `/workspace/SAGE/client/room_descs/`

On your local machine:
- RunPod state JSON under `analysis_outputs/runpod_sage_baseline_*.json` (no secrets)

## Notes / Failure Modes

- **Docker-in-RunPod**: Pod B bootstraps Docker and uses `nvidia-container-toolkit` for `--gpus all`. If the pod template disallows starting `dockerd`, Isaac Sim cannot run. In that case, we will need to switch Pod B to an Isaac Sim *archive install* flow instead of Docker.
- **Costs**: Pods keep billing until terminated. Use `--keep-pods` only if you plan to SSH back in and debug.
- **SLURM_JOB_ID**: SAGE’s MCP socket port is derived from `SLURM_JOB_ID`. The automation pins it to `runpod_sage_001` by default.

