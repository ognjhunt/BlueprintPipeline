#!/usr/bin/env bash
set -euo pipefail
# =============================================================================
# Run the RGB rendering test on the T4 GPU VM.
#
# This script runs FROM your Mac. It:
#   1. Starts the T4 VM (if stopped)
#   2. Starts Xorg + Docker container on the VM
#   3. Waits for gRPC readiness
#   4. Runs a single-scene pipeline with RGB capture ENABLED
#   5. Checks results for non-black frames
#   6. Downloads episode data to your Mac
#   7. Stops the VM (to save money)
#
# Prerequisites:
#   - VM created: bash scripts/create-t4-test-vm.sh
#   - VM provisioned: bash scripts/setup-t4-vm.sh
#
# Usage:
#   bash scripts/run-t4-rgb-test.sh
#
# Cost: ~$0.95/hr. Total test ~$0.70-1.50 depending on duration.
# =============================================================================

VM_NAME="${T4_VM_NAME:-geniesim-t4-test}"
VM_ZONE="${T4_VM_ZONE:-asia-east1-c}"
SCENE_DIR="${T4_TEST_SCENE:-test_scenes/scenes/lightwheel_kitchen}"
LOCAL_RESULTS_DIR="${T4_RESULTS_DIR:-./t4_rgb_test_results}"
AUTO_STOP="${T4_AUTO_STOP:-1}"  # Set to 0 to keep VM running after test

ssh_cmd() {
  gcloud compute ssh "${VM_NAME}" --zone="${VM_ZONE}" -- "$@"
}

echo "============================================================"
echo " T4 RGB Rendering Test"
echo "============================================================"
echo "VM:      ${VM_NAME} (${VM_ZONE})"
echo "Scene:   ${SCENE_DIR}"
echo "Results: ${LOCAL_RESULTS_DIR}"
echo "============================================================"

# ── Step 1: Start VM ─────────────────────────────────────────────────────────
echo "[t4-test] Step 1/7: Starting VM..."
_status=$(gcloud compute instances describe "${VM_NAME}" --zone="${VM_ZONE}" --format="value(status)" 2>/dev/null || echo "NOT_FOUND")

case "${_status}" in
  RUNNING)
    echo "[t4-test] VM already running."
    ;;
  TERMINATED|STOPPED)
    echo "[t4-test] Starting stopped VM..."
    gcloud compute instances start "${VM_NAME}" --zone="${VM_ZONE}"
    echo "[t4-test] Waiting for VM to boot..."
    sleep 30
    ;;
  *)
    echo "[t4-test] ERROR: VM '${VM_NAME}' not found or in unexpected state: ${_status}" >&2
    echo "[t4-test] Run: bash scripts/create-t4-test-vm.sh" >&2
    exit 1
    ;;
esac

# ── Step 2: Wait for SSH ─────────────────────────────────────────────────────
echo "[t4-test] Step 2/7: Waiting for SSH..."
for i in $(seq 1 30); do
  if ssh_cmd "echo 'SSH ready'" 2>/dev/null; then
    break
  fi
  if [ "$i" -eq 30 ]; then
    echo "[t4-test] ERROR: SSH not ready after 5 minutes" >&2
    exit 1
  fi
  echo "  Attempt $i/30..."
  sleep 10
done

# Quick GPU check
echo "[t4-test] GPU status:"
ssh_cmd nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# ── Step 3: Start Xorg + Docker container ─────────────────────────────────────
echo "[t4-test] Step 3/7: Starting Xorg display + GenieSim container..."
ssh_cmd bash -c "'
  cd ~/BlueprintPipeline
  # Ensure we have the RGB test config
  if [ ! -f configs/realism_rgb_test.env ]; then
    echo \"ERROR: configs/realism_rgb_test.env not found. Run git pull first.\" >&2
    exit 1
  fi

  # Source the RGB test config for container env vars
  set -a
  source configs/realism_rgb_test.env
  set +a
  export ENABLE_CAMERAS=1

  # Start Xorg + container
  bash scripts/vm-start.sh
'"

# ── Step 4: Wait for gRPC readiness ──────────────────────────────────────────
echo "[t4-test] Step 4/7: Waiting for gRPC server..."
_grpc_ready=0
for i in $(seq 1 60); do
  if ssh_cmd bash -c "'
    PYTHONPATH=\"\" python3 -c \"
import grpc, sys
ch = grpc.insecure_channel(\\\"localhost:50051\\\")
grpc.channel_ready_future(ch).result(timeout=2)
sys.exit(0)
\" 2>/dev/null
  '" 2>/dev/null; then
    _grpc_ready=1
    echo "[t4-test] gRPC ready after ~$((i * 5))s"
    break
  fi
  echo "  Waiting for gRPC... (attempt $i/60)"
  sleep 5
done

if [ "${_grpc_ready}" = "0" ]; then
  echo "[t4-test] ERROR: gRPC not ready after 5 minutes." >&2
  echo "[t4-test] Check server logs: gcloud compute ssh ${VM_NAME} --zone=${VM_ZONE} -- 'sudo docker logs geniesim-server --tail 50'" >&2
  exit 1
fi

# ── Step 5: Run pipeline with RGB enabled ─────────────────────────────────────
echo "[t4-test] Step 5/7: Running pipeline with RGB capture ENABLED..."
echo "[t4-test] This may take 30-60 minutes for a full scene."
echo ""

REMOTE_LOG="/tmp/t4_rgb_test.log"

ssh_cmd bash -c "'
  cd ~/BlueprintPipeline

  # Source RGB test config
  set -a
  source configs/realism_rgb_test.env
  set +a

  # Fixed env vars
  export PYTHONPATH=\"\${HOME}/BlueprintPipeline:\${HOME}/BlueprintPipeline/episode-generation-job:\${PYTHONPATH:-}\"
  export GENIESIM_HOST=localhost
  export GENIESIM_PORT=50051
  export GENIESIM_SKIP_DEFAULT_LIGHTING=1
  export ENABLE_CAMERAS=1

  echo \"[t4-test] Config: SKIP_RGB_CAPTURE=\${SKIP_RGB_CAPTURE}, ENABLE_CAMERAS=\${ENABLE_CAMERAS}\"
  echo \"[t4-test] Starting pipeline...\"

  python3 tools/run_local_pipeline.py \
    --scene-dir ./${SCENE_DIR} \
    --steps genie-sim-submit \
    --force-rerun genie-sim-submit \
    --use-geniesim --fail-fast \
    2>&1 | tee ${REMOTE_LOG}
'"
PIPELINE_EXIT=$?

echo ""
echo "[t4-test] Pipeline exited with code: ${PIPELINE_EXIT}"

# ── Step 6: Check results ─────────────────────────────────────────────────────
echo "[t4-test] Step 6/7: Checking RGB capture results..."
ssh_cmd bash -c "'
  cd ~/BlueprintPipeline

  echo \"\"
  echo \"=== RGB Capture Check ===\"
  python3 -c \"
import json, glob, os

recording_dir = os.path.expanduser(\\\"~/BlueprintPipeline/${SCENE_DIR}/geniesim/recordings\\\")
json_files = sorted(glob.glob(os.path.join(recording_dir, \\\"*.json\\\")))

if not json_files:
    print(\\\"WARNING: No episode JSON files found in\\\", recording_dir)
else:
    total_camera_frames = 0
    total_episodes = 0
    for f in json_files:
        with open(f) as fh:
            data = json.load(fh)
        meta = data.get(\\\"metadata\\\", {})
        quality = meta.get(\\\"quality_breakdown\\\", {})
        camera = quality.get(\\\"camera_frames\\\", {})
        captured = camera.get(\\\"camera_capture_frames\\\", 0)
        total_frames = camera.get(\\\"total_frames\\\", 0)
        confidence = camera.get(\\\"confidence\\\", 0)
        score = meta.get(\\\"quality_score\\\", 0)

        total_camera_frames += captured
        total_episodes += 1

        status = \\\"OK\\\" if captured > 0 else \\\"FAIL (black frames)\\\"
        print(f\\\"  Episode {os.path.basename(f)}: camera_frames={captured}/{total_frames}, confidence={confidence:.2f}, score={score:.3f} [{status}]\\\")

    print(f\\\"\\\")
    if total_camera_frames > 0:
        print(f\\\"SUCCESS: {total_camera_frames} RGB frames captured across {total_episodes} episodes!\\\")
        print(f\\\"T4 GPU renders RGB correctly. Ready to use for Stage 4.\\\")
    else:
        print(f\\\"FAIL: 0 RGB frames across {total_episodes} episodes.\\\")
        print(f\\\"T4 also renders black frames. Try different renderer or Isaac Sim version.\\\")
\"
'"

# ── Step 7: Download results ──────────────────────────────────────────────────
echo "[t4-test] Step 7/7: Downloading results to ${LOCAL_RESULTS_DIR}..."
mkdir -p "${LOCAL_RESULTS_DIR}"

# Download episode JSONs (individual files to avoid gcloud scp --recurse nesting bug)
_remote_rec_dir="~/BlueprintPipeline/${SCENE_DIR}/geniesim/recordings"
_rec_files=$(ssh_cmd "ls ${_remote_rec_dir}/ 2>/dev/null" 2>/dev/null || true)
if [ -n "${_rec_files}" ]; then
  for _f in ${_rec_files}; do
    gcloud compute scp \
      "${VM_NAME}:${_remote_rec_dir}/${_f}" \
      "${LOCAL_RESULTS_DIR}/${_f}" \
      --zone="${VM_ZONE}" 2>/dev/null || true
  done
  echo "[t4-test] Downloaded $(echo "${_rec_files}" | wc -w | tr -d ' ') recording files."
else
  echo "[t4-test] No recording files to download."
fi

# Download log
gcloud compute scp \
  "${VM_NAME}:${REMOTE_LOG}" \
  "${LOCAL_RESULTS_DIR}/t4_rgb_test.log" \
  --zone="${VM_ZONE}" 2>/dev/null || echo "[t4-test] No log file to download."

echo "[t4-test] Results saved to: ${LOCAL_RESULTS_DIR}"

# ── Auto-stop VM ──────────────────────────────────────────────────────────────
if [ "${AUTO_STOP}" = "1" ]; then
  echo ""
  echo "[t4-test] Stopping VM to save money..."
  gcloud compute instances stop "${VM_NAME}" --zone="${VM_ZONE}"
  echo "[t4-test] VM stopped."
else
  echo ""
  echo "[t4-test] VM still running. Remember to stop it:"
  echo "  gcloud compute instances stop ${VM_NAME} --zone=${VM_ZONE}"
fi

echo ""
echo "============================================================"
echo " T4 RGB Test Complete"
echo "============================================================"
echo "Results: ${LOCAL_RESULTS_DIR}"
echo "Pipeline exit code: ${PIPELINE_EXIT}"
echo "============================================================"

exit "${PIPELINE_EXIT}"
