#!/usr/bin/env bash
set -euo pipefail
# =============================================================================
# Run the RGB rendering test on a RunPod L40S pod via SSH.
#
# This connects to an already-running RunPod pod and:
#   1. Provisions it (drivers, Docker, Xorg, repo, Docker image)
#   2. Starts Xorg + GenieSim container
#   3. Runs the RGB test pipeline
#   4. Downloads results to your Mac
#
# Usage:
#   bash scripts/run-runpod-rgb-test.sh <ssh-host> <ssh-port>
#
# Example:
#   bash scripts/run-runpod-rgb-test.sh 194.26.196.42 22188
#
# The SSH host/port can be found on the RunPod dashboard under "Connect".
# =============================================================================

if [ $# -lt 2 ]; then
  echo "Usage: bash scripts/run-runpod-rgb-test.sh <ssh-host> <ssh-port>"
  echo ""
  echo "Get the SSH details from: https://www.runpod.io/console/pods → Connect"
  exit 1
fi

RUNPOD_HOST="$1"
RUNPOD_PORT="$2"
RUNPOD_USER="${RUNPOD_USER:-root}"
LOCAL_RESULTS_DIR="${L40S_RESULTS_DIR:-./l40s_rgb_test_results}"
REPO_BRANCH="${REPO_BRANCH:-main}"
REPO_URL="${REPO_URL:-$(git remote get-url origin 2>/dev/null || echo 'https://github.com/your-org/BlueprintPipeline')}"
NGC_API_KEY="${NGC_API_KEY:-}"
SCENE_DIR="${L40S_TEST_SCENE:-test_scenes/scenes/lightwheel_kitchen}"

rpod_ssh() {
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
    -p "${RUNPOD_PORT}" "${RUNPOD_USER}@${RUNPOD_HOST}" "$@"
}

rpod_scp_from() {
  scp -o StrictHostKeyChecking=no -P "${RUNPOD_PORT}" \
    "${RUNPOD_USER}@${RUNPOD_HOST}:$1" "$2"
}

echo "============================================================"
echo " L40S RGB Rendering Test (RunPod)"
echo "============================================================"
echo "Host:    ${RUNPOD_HOST}:${RUNPOD_PORT}"
echo "Scene:   ${SCENE_DIR}"
echo "Results: ${LOCAL_RESULTS_DIR}"
echo "============================================================"

# ── Step 1: Verify connection + GPU ──────────────────────────────────────────
echo "[l40s] Step 1/7: Verifying SSH + GPU..."
rpod_ssh "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"

# ── Step 2: Install dependencies ─────────────────────────────────────────────
echo "[l40s] Step 2/7: Installing dependencies..."
rpod_ssh bash -c "'
  apt-get update -qq 2>/dev/null
  apt-get install -y -qq git xserver-xorg-core 2>/dev/null || true

  # NVIDIA Container Toolkit
  if ! command -v nvidia-container-cli &>/dev/null; then
    echo \"Installing nvidia-container-toolkit...\"
    distribution=\$(. /etc/os-release; echo \$ID\$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null
    curl -s -L \"https://nvidia.github.io/libnvidia-container/\${distribution}/libnvidia-container.list\" | \
      sed \"s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g\" | \
      tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    apt-get update -qq && apt-get install -y -qq nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker 2>/dev/null || service docker restart 2>/dev/null || true
  fi
  echo \"Dependencies ready.\"
'"

# ── Step 3: Clone repo ──────────────────────────────────────────────────────
echo "[l40s] Step 3/7: Setting up repo..."
rpod_ssh bash -c "'
  cd /workspace
  if [ -d BlueprintPipeline/.git ]; then
    echo \"Repo exists, pulling latest...\"
    cd BlueprintPipeline && git pull origin ${REPO_BRANCH} || true
  else
    echo \"Cloning repo...\"
    git clone --branch ${REPO_BRANCH} ${REPO_URL} BlueprintPipeline
    cd BlueprintPipeline
  fi
  echo \"Repo ready.\"
'"

# ── Step 4: NGC auth + Docker build ──────────────────────────────────────────
echo "[l40s] Step 4/7: Building Docker image..."

# NGC auth if key provided
if [ -n "${NGC_API_KEY}" ]; then
  echo "[l40s] Authenticating with NGC..."
  rpod_ssh "echo '${NGC_API_KEY}' | docker login nvcr.io -u '\$oauthtoken' --password-stdin" 2>/dev/null
else
  echo "[l40s] NOTE: NGC_API_KEY not set. If the build fails, set it:"
  echo "  export NGC_API_KEY=<your-key> && bash scripts/run-runpod-rgb-test.sh ${RUNPOD_HOST} ${RUNPOD_PORT}"
fi

rpod_ssh bash -c "'
  cd /workspace/BlueprintPipeline
  if docker images geniesim-server:latest --format \"{{.ID}}\" 2>/dev/null | head -1 | grep -q .; then
    echo \"Docker image already exists.\"
  else
    echo \"Building geniesim-server (nocurobo variant, ~20 min)...\"
    docker build -f Dockerfile.geniesim-server-nocurobo -t geniesim-server:latest . 2>&1 | tail -10
  fi
  echo \"Docker image ready.\"
'"

# ── Step 5: Start Xorg + GenieSim container ──────────────────────────────────
echo "[l40s] Step 5/7: Starting Xorg + container..."
rpod_ssh bash -c "'
  cd /workspace/BlueprintPipeline
  set -a
  source configs/realism_rgb_test.env
  set +a
  export ENABLE_CAMERAS=1
  bash scripts/vm-start.sh
'"

# ── Step 6: Wait for gRPC + run test ─────────────────────────────────────────
echo "[l40s] Step 6/7: Waiting for gRPC + running pipeline..."

# Wait for gRPC
echo "[l40s] Polling gRPC..."
_grpc_ready=0
for i in $(seq 1 90); do
  if rpod_ssh bash -c "'
    PYTHONPATH=\"\" python3 -c \"
import grpc, sys
ch = grpc.insecure_channel(\\\"localhost:50051\\\")
grpc.channel_ready_future(ch).result(timeout=2)
sys.exit(0)
\" 2>/dev/null
  '" 2>/dev/null; then
    _grpc_ready=1
    echo "[l40s] gRPC ready after ~$((i * 5))s"
    break
  fi
  echo "  Waiting... (attempt $i/90)"
  sleep 5
done

if [ "${_grpc_ready}" = "0" ]; then
  echo "[l40s] ERROR: gRPC not ready after 7.5 minutes" >&2
  echo "[l40s] Check server logs:"
  echo "  ssh -p ${RUNPOD_PORT} ${RUNPOD_USER}@${RUNPOD_HOST} 'docker logs geniesim-server --tail 50'"
  exit 1
fi

# Run the pipeline
echo "[l40s] Running pipeline with RGB capture..."
rpod_ssh bash -c "'
  cd /workspace/BlueprintPipeline
  set -a
  source configs/realism_rgb_test.env
  set +a
  export PYTHONPATH=\"/workspace/BlueprintPipeline:/workspace/BlueprintPipeline/episode-generation-job:\${PYTHONPATH:-}\"
  export GENIESIM_HOST=localhost
  export GENIESIM_PORT=50051
  export GENIESIM_SKIP_DEFAULT_LIGHTING=1
  export ENABLE_CAMERAS=1

  python3 tools/run_local_pipeline.py \
    --scene-dir ./${SCENE_DIR} \
    --steps genie-sim-submit \
    --force-rerun genie-sim-submit \
    --use-geniesim --fail-fast \
    2>&1 | tee /tmp/l40s_rgb_test.log
'"
PIPELINE_EXIT=$?

# ── Step 7: Check + download results ─────────────────────────────────────────
echo "[l40s] Step 7/7: Checking results + downloading..."
mkdir -p "${LOCAL_RESULTS_DIR}"

rpod_ssh bash -c "'
  cd /workspace/BlueprintPipeline
  python3 -c \"
import json, glob, os, sys
recording_dir = \\\"${SCENE_DIR}/geniesim/recordings\\\"
json_files = sorted(glob.glob(os.path.join(recording_dir, \\\"*.json\\\")))
if not json_files:
    print(\\\"WARNING: No episode JSON files found\\\")
    sys.exit(1)
total_camera_frames = 0
for f in json_files:
    with open(f) as fh:
        data = json.load(fh)
    meta = data.get(\\\"metadata\\\", {})
    quality = meta.get(\\\"quality_breakdown\\\", {})
    camera = quality.get(\\\"camera_frames\\\", {})
    captured = camera.get(\\\"camera_capture_frames\\\", 0)
    score = meta.get(\\\"quality_score\\\", 0)
    total_camera_frames += captured
    status = \\\"OK\\\" if captured > 0 else \\\"FAIL\\\"
    print(f\\\"  {os.path.basename(f)}: camera={captured}, score={score:.3f} [{status}]\\\")
print()
if total_camera_frames > 0:
    print(f\\\"SUCCESS: {total_camera_frames} RGB frames captured on L40S!\\\")
    print(f\\\"L40S is the production GPU for Stage 4.\\\")
else:
    print(f\\\"FAIL: 0 RGB frames. Server logs:\\\")
    import subprocess
    subprocess.run([\\\"docker\\\", \\\"logs\\\", \\\"geniesim-server\\\", \\\"--tail\\\", \\\"30\\\"], capture_output=False)
\"
'"

# Download results
echo "[l40s] Downloading results..."
rpod_scp_from "/workspace/BlueprintPipeline/${SCENE_DIR}/geniesim/recordings/*" "${LOCAL_RESULTS_DIR}/" 2>/dev/null || true
rpod_scp_from "/tmp/l40s_rgb_test.log" "${LOCAL_RESULTS_DIR}/l40s_rgb_test.log" 2>/dev/null || true

echo ""
echo "============================================================"
echo " L40S RGB Test Complete"
echo "============================================================"
echo "Results: ${LOCAL_RESULTS_DIR}"
echo "Pipeline exit code: ${PIPELINE_EXIT}"
echo ""
echo "IMPORTANT: Stop the RunPod pod to save money!"
echo "  runpodctl stop pod <pod-id>"
echo "  Or: https://www.runpod.io/console/pods"
echo "============================================================"
