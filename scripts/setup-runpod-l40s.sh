#!/usr/bin/env bash
set -euo pipefail
# =============================================================================
# Fallback 2: Provision an L40S GPU on RunPod for RGB rendering.
#
# If all renderer modes fail on T4, the L40S (Ada Lovelace, 48GB VRAM,
# 142 RT cores) is NVIDIA's recommended GPU for Isaac Sim.
#
# This script:
#   1. Creates an L40S pod on RunPod using the CLI
#   2. Installs dependencies (NVIDIA toolkit, Docker, Xorg)
#   3. Clones the repo and builds the Docker image
#   4. Runs the RGB test
#
# Prerequisites:
#   - RunPod CLI installed: pip install runpodctl
#   - RunPod API key set: export RUNPOD_API_KEY=<key>
#     OR: runpodctl config --apikey <key>
#   - SSH key configured in RunPod dashboard
#
# Cost: ~$1.50-2.50/hr for L40S. Total test: ~$3-5.
#
# Usage:
#   bash scripts/setup-runpod-l40s.sh
#
# Alternative providers (if RunPod is unavailable):
#   Lambda Cloud: https://cloud.lambdalabs.com (L40S ~$1.50/hr)
#   Vast.ai: https://vast.ai (L40S from ~$0.80/hr spot)
# =============================================================================

RUNPOD_GPU="${RUNPOD_GPU:-NVIDIA L40S}"
RUNPOD_IMAGE="${RUNPOD_IMAGE:-runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04}"
RUNPOD_DISK="${RUNPOD_DISK:-100}"
RUNPOD_VOLUME="${RUNPOD_VOLUME:-100}"
RUNPOD_NAME="${RUNPOD_NAME:-blueprint-l40s-rgb-test}"
REPO_BRANCH="${REPO_BRANCH:-main}"
LOCAL_RESULTS_DIR="${L40S_RESULTS_DIR:-./l40s_rgb_test_results}"

# GitHub repo URL
REPO_URL="${REPO_URL:-$(git remote get-url origin 2>/dev/null || echo 'https://github.com/your-org/BlueprintPipeline')}"

echo "============================================================"
echo " Fallback 2: L40S RGB Test on RunPod"
echo "============================================================"
echo "GPU:       ${RUNPOD_GPU}"
echo "Pod Name:  ${RUNPOD_NAME}"
echo "Image:     ${RUNPOD_IMAGE}"
echo "Disk:      ${RUNPOD_DISK}GB container + ${RUNPOD_VOLUME}GB volume"
echo "Repo:      ${REPO_URL} (${REPO_BRANCH})"
echo "Results:   ${LOCAL_RESULTS_DIR}"
echo "Est. cost: ~\$1.50-2.50/hr"
echo "============================================================"
echo ""

# ── Pre-flight checks ────────────────────────────────────────────────────────
_preflight_ok=true

if ! command -v runpodctl &>/dev/null; then
  echo "[l40s] ERROR: runpodctl not installed."
  echo "  Install: pip install runpodctl"
  echo "  Docs: https://docs.runpod.io/cli/installation"
  _preflight_ok=false
fi

if [ -z "${RUNPOD_API_KEY:-}" ]; then
  # Check if runpodctl has a saved config
  if ! runpodctl config 2>/dev/null | grep -q "apiKey"; then
    echo "[l40s] WARNING: RUNPOD_API_KEY not set and no saved config."
    echo "  Set: export RUNPOD_API_KEY=<your-api-key>"
    echo "  Or:  runpodctl config --apikey <your-api-key>"
    echo ""
    echo "  Get your API key from: https://www.runpod.io/console/user/settings"
    _preflight_ok=false
  fi
fi

if [ "${_preflight_ok}" = "false" ]; then
  echo ""
  echo "[l40s] Fix the above issues and re-run this script."
  exit 1
fi

# ── Step 1: Create the pod ───────────────────────────────────────────────────
echo "[l40s] Step 1/5: Creating L40S pod on RunPod..."

# Check if pod already exists
_existing=$(runpodctl get pod 2>/dev/null | grep "${RUNPOD_NAME}" | head -1 || true)
if [ -n "${_existing}" ]; then
  POD_ID=$(echo "${_existing}" | awk '{print $1}')
  echo "[l40s] Pod already exists: ${POD_ID}"
  echo "[l40s] Status: $(echo "${_existing}" | awk '{print $3}')"
else
  echo "[l40s] Creating new pod..."
  # Create pod with L40S GPU
  POD_ID=$(runpodctl create pod \
    --name "${RUNPOD_NAME}" \
    --gpuType "${RUNPOD_GPU}" \
    --gpuCount 1 \
    --imageName "${RUNPOD_IMAGE}" \
    --containerDiskSize "${RUNPOD_DISK}" \
    --volumeSize "${RUNPOD_VOLUME}" \
    --ports "22/tcp,50051/tcp" \
    --env "NVIDIA_DRIVER_CAPABILITIES=all" \
    2>&1 | grep -oP 'pod_\w+' | head -1 || true)

  if [ -z "${POD_ID}" ]; then
    echo "[l40s] ERROR: Failed to create pod. Check RunPod dashboard."
    echo "[l40s] You may need to create it manually at https://www.runpod.io/console/pods"
    echo ""
    echo "[l40s] Manual setup instructions:"
    echo "  1. Select GPU: NVIDIA L40S (48GB)"
    echo "  2. Template: RunPod PyTorch 2.2.1"
    echo "  3. Container disk: ${RUNPOD_DISK}GB, Volume: ${RUNPOD_VOLUME}GB"
    echo "  4. Expose ports: 22/tcp, 50051/tcp"
    echo "  5. Then run: bash scripts/run-runpod-rgb-test.sh <pod-ip>"
    exit 1
  fi
  echo "[l40s] Pod created: ${POD_ID}"
fi

# ── Step 2: Wait for pod to be ready ─────────────────────────────────────────
echo "[l40s] Step 2/5: Waiting for pod to be running..."
for i in $(seq 1 60); do
  _pod_status=$(runpodctl get pod 2>/dev/null | grep "${POD_ID}" | awk '{print $3}' || echo "UNKNOWN")
  if [ "${_pod_status}" = "RUNNING" ]; then
    echo "[l40s] Pod is running!"
    break
  fi
  if [ "$i" -eq 60 ]; then
    echo "[l40s] ERROR: Pod not ready after 5 minutes. Status: ${_pod_status}" >&2
    exit 1
  fi
  echo "  Status: ${_pod_status} (attempt $i/60)"
  sleep 5
done

# Get SSH connection info
echo "[l40s] Getting SSH connection details..."
_pod_info=$(runpodctl get pod 2>/dev/null | grep "${POD_ID}" || true)
echo "[l40s] Pod info: ${_pod_info}"

echo ""
echo "============================================================"
echo " Pod Created — Manual SSH Setup Required"
echo "============================================================"
echo ""
echo "RunPod pods use dynamic SSH ports. To connect:"
echo "  1. Go to https://www.runpod.io/console/pods"
echo "  2. Click 'Connect' on pod '${RUNPOD_NAME}'"
echo "  3. Copy the SSH command (e.g., ssh root@<ip> -p <port>)"
echo "  4. Then run the setup + test:"
echo ""
echo "  bash scripts/run-runpod-rgb-test.sh <pod-ssh-host> <pod-ssh-port>"
echo ""
echo "Or copy-paste this inside the pod's terminal/Jupyter:"
echo "============================================================"

cat <<'SETUP_SCRIPT'

# === Run this inside the RunPod pod ===

# 1. Install system dependencies
apt-get update -qq && apt-get install -y -qq \
  git xserver-xorg-core xserver-xorg-video-nvidia \
  libnvidia-gl-550 2>/dev/null || true

# 2. Verify GPU
nvidia-smi
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# 3. Install NVIDIA Container Toolkit (if not already)
if ! command -v nvidia-container-cli &>/dev/null; then
  distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L "https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list" | \
    sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -qq && apt-get install -y -qq nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker 2>/dev/null || service docker restart
fi

# 4. Clone repo
cd /workspace
if [ -d BlueprintPipeline/.git ]; then
  cd BlueprintPipeline && git pull origin main
else
  git clone --branch main <YOUR_REPO_URL> BlueprintPipeline
  cd BlueprintPipeline
fi

# 5. NGC login (required for Isaac Sim base image)
echo "You need NGC credentials for Isaac Sim:"
echo "  docker login nvcr.io -u '\$oauthtoken' -p '<your-ngc-api-key>'"

# 6. Build Docker image (use nocurobo for speed when running with cuRobo disabled)
docker build -f Dockerfile.geniesim-server-nocurobo -t geniesim-server:latest . 2>&1 | tail -10

# 7. Start Xorg + container
set -a && source configs/realism_rgb_test.env && set +a
export ENABLE_CAMERAS=1
bash scripts/vm-start.sh

# 8. Wait for gRPC
echo "Waiting for gRPC..."
for i in $(seq 1 60); do
  if PYTHONPATH="" python3 -c "
import grpc, sys
ch = grpc.insecure_channel('localhost:50051')
grpc.channel_ready_future(ch).result(timeout=2)
sys.exit(0)
" 2>/dev/null; then
    echo "gRPC ready!"
    break
  fi
  sleep 5
done

# 9. Run the test
export PYTHONPATH="/workspace/BlueprintPipeline:/workspace/BlueprintPipeline/episode-generation-job:${PYTHONPATH:-}"
export GENIESIM_HOST=localhost GENIESIM_PORT=50051 GENIESIM_SKIP_DEFAULT_LIGHTING=1

python3 tools/run_local_pipeline.py \
  --scene-dir ./test_scenes/scenes/lightwheel_kitchen \
  --steps genie-sim-submit \
  --force-rerun genie-sim-submit \
  --use-geniesim --fail-fast 2>&1 | tee /tmp/l40s_rgb_test.log

# 10. Check results
python3 -c "
import json, glob, os
recording_dir = 'test_scenes/scenes/lightwheel_kitchen/geniesim/recordings'
json_files = sorted(glob.glob(os.path.join(recording_dir, '*.json')))
if not json_files:
    print('WARNING: No episode JSON files found')
else:
    total_camera_frames = 0
    for f in json_files:
        with open(f) as fh:
            data = json.load(fh)
        meta = data.get('metadata', {})
        quality = meta.get('quality_breakdown', {})
        camera = quality.get('camera_frames', {})
        captured = camera.get('camera_capture_frames', 0)
        total_camera_frames += captured
        score = meta.get('quality_score', 0)
        print(f'  {os.path.basename(f)}: camera_frames={captured}, score={score:.3f}')
    if total_camera_frames > 0:
        print(f'\\nSUCCESS: {total_camera_frames} RGB frames captured!')
        print('L40S renders correctly. Use RunPod L40S for Stage 4.')
    else:
        print('\\nFAIL: 0 RGB frames. Check server logs:')
        print('  docker logs geniesim-server 2>&1 | grep -i camera | tail -20')
"
SETUP_SCRIPT

echo ""
echo "============================================================"
echo " Pod ID: ${POD_ID:-<check RunPod dashboard>}"
echo " Remember to STOP the pod when done to save money!"
echo "   runpodctl stop pod ${POD_ID:-<pod-id>}"
echo "   Or use the RunPod dashboard."
echo "============================================================"
