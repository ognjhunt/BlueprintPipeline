#!/usr/bin/env bash
set -euo pipefail
# =============================================================================
# Fallback 1: Try different renderer modes on the SAME T4 VM.
#
# If the default RaytracedLighting mode produces black frames, this script
# cycles through alternative renderer modes:
#   1. RealTimePathTracing
#   2. RaytracedLighting + RT2 enabled
#   3. RealTimePathTracing + RT2 enabled
#
# The patches now read BP_RENDERER_MODE and BP_RT2_ENABLED env vars at runtime,
# so switching modes only requires restarting the container (no rebuild needed).
#
# Runs FROM your Mac. The T4 VM must already be running with setup complete.
#
# Usage:
#   bash scripts/try-renderer-modes.sh
#
# Cost: ~$0.30-0.50 per mode (container restart + short test, no rebuild)
# =============================================================================

VM_NAME="${T4_VM_NAME:-geniesim-t4-test}"
VM_ZONE="${T4_VM_ZONE:-asia-east1-c}"
SCENE_DIR="${T4_TEST_SCENE:-test_scenes/scenes/lightwheel_kitchen}"
LOCAL_RESULTS_DIR="${T4_RESULTS_DIR:-./t4_renderer_mode_results}"
# Renderer modes to try, in order
MODES="${RENDERER_MODES:-RealTimePathTracing RaytracedLighting+RT2 RealTimePathTracing+RT2}"

ssh_cmd() {
  gcloud compute ssh "${VM_NAME}" --zone="${VM_ZONE}" -- "$@"
}

echo "============================================================"
echo " Fallback 1: Renderer Mode Sweep on T4"
echo "============================================================"
echo "VM:      ${VM_NAME} (${VM_ZONE})"
echo "Scene:   ${SCENE_DIR}"
echo "Modes:   ${MODES}"
echo "Results: ${LOCAL_RESULTS_DIR}"
echo "============================================================"

# ── Verify VM is running ──────────────────────────────────────────────────────
_status=$(gcloud compute instances describe "${VM_NAME}" --zone="${VM_ZONE}" --format="value(status)" 2>/dev/null || echo "NOT_FOUND")
if [ "${_status}" != "RUNNING" ]; then
  echo "[renderer-modes] ERROR: VM '${VM_NAME}' is ${_status}. Start it first:" >&2
  echo "  gcloud compute instances start ${VM_NAME} --zone=${VM_ZONE}" >&2
  exit 1
fi

# Wait for SSH
echo "[renderer-modes] Checking SSH..."
for i in $(seq 1 10); do
  if ssh_cmd "echo 'SSH ready'" 2>/dev/null; then break; fi
  sleep 5
done

mkdir -p "${LOCAL_RESULTS_DIR}"

_success_mode=""
_mode_num=0

for MODE in ${MODES}; do
  _mode_num=$((_mode_num + 1))
  echo ""
  echo "============================================================"
  echo " Mode ${_mode_num}: ${MODE}"
  echo "============================================================"

  # Parse mode string: "RealTimePathTracing+RT2" → renderer=RealTimePathTracing, rt2=1
  RENDERER_MODE="${MODE%%+*}"
  RT2_FLAG="0"
  if [[ "${MODE}" == *"+RT2"* ]]; then
    RT2_FLAG="1"
  fi

  echo "[renderer-modes] Renderer: ${RENDERER_MODE}, RT2: ${RT2_FLAG}"

  # ── Step 1: Restart container with new env vars ────────────────────────────
  # The patches read BP_RENDERER_MODE and BP_RT2_ENABLED at runtime,
  # so we just need to restart the container with new env vars.
  echo "[renderer-modes] Restarting container with new renderer mode..."
  ssh_cmd bash -c "'
    cd ~/BlueprintPipeline

    # Stop existing container
    sudo docker stop geniesim-server 2>/dev/null || true
    sudo docker rm geniesim-server 2>/dev/null || true

    # Source base config and override renderer settings
    set -a
    source configs/realism_rgb_test.env
    set +a
    export ENABLE_CAMERAS=1
    export BP_RENDERER_MODE=${RENDERER_MODE}
    export BP_RT2_ENABLED=${RT2_FLAG}

    echo \"Starting with BP_RENDERER_MODE=\${BP_RENDERER_MODE}, BP_RT2_ENABLED=\${BP_RT2_ENABLED}\"

    # Start Xorg + container
    bash scripts/vm-start.sh
  '"

  # ── Step 2: Wait for gRPC ──────────────────────────────────────────────────
  echo "[renderer-modes] Waiting for gRPC..."
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
      echo "[renderer-modes] gRPC ready."
      break
    fi
    sleep 5
  done

  if [ "${_grpc_ready}" = "0" ]; then
    echo "[renderer-modes] gRPC not ready for mode ${MODE}. Skipping."
    gcloud compute scp "${VM_NAME}:/tmp/geniesim_server.log" \
      "${LOCAL_RESULTS_DIR}/server_log_${MODE//+/_}.txt" \
      --zone="${VM_ZONE}" 2>/dev/null || true
    continue
  fi

  # ── Step 3: Run short test (1 task only for speed) ─────────────────────────
  echo "[renderer-modes] Running quick RGB test (1 task)..."
  _test_log="${LOCAL_RESULTS_DIR}/test_${MODE//+/_}.log"

  ssh_cmd bash -c "'
    cd ~/BlueprintPipeline
    set -a
    source configs/realism_rgb_test.env
    set +a
    export PYTHONPATH=\"\${HOME}/BlueprintPipeline:\${HOME}/BlueprintPipeline/episode-generation-job:\${PYTHONPATH:-}\"
    export GENIESIM_HOST=localhost
    export GENIESIM_PORT=50051
    export GENIESIM_SKIP_DEFAULT_LIGHTING=1
    export ENABLE_CAMERAS=1
    export BP_RENDERER_MODE=${RENDERER_MODE}
    export BP_RT2_ENABLED=${RT2_FLAG}

    # Run just 1 task for quick validation
    export GENIESIM_MAX_TASKS=1

    python3 tools/run_local_pipeline.py \
      --scene-dir ./${SCENE_DIR} \
      --steps genie-sim-submit \
      --force-rerun genie-sim-submit \
      --use-geniesim --fail-fast \
      2>&1
  '" > "${_test_log}" 2>&1 || true

  # ── Step 4: Check results ──────────────────────────────────────────────────
  echo "[renderer-modes] Checking RGB capture for mode ${MODE}..."
  _rgb_ok=$(ssh_cmd bash -c "'
    cd ~/BlueprintPipeline
    python3 -c \"
import json, glob, os, sys
recording_dir = os.path.expanduser(\\\"~/BlueprintPipeline/${SCENE_DIR}/geniesim/recordings\\\")
json_files = sorted(glob.glob(os.path.join(recording_dir, \\\"*.json\\\")))
if not json_files:
    print(\\\"NO_DATA\\\")
    sys.exit(1)
total_camera_frames = 0
for f in json_files:
    with open(f) as fh:
        data = json.load(fh)
    meta = data.get(\\\"metadata\\\", {})
    quality = meta.get(\\\"quality_breakdown\\\", {})
    camera = quality.get(\\\"camera_frames\\\", {})
    captured = camera.get(\\\"camera_capture_frames\\\", 0)
    total_camera_frames += captured
if total_camera_frames > 0:
    print(f\\\"SUCCESS:{total_camera_frames}\\\")
else:
    print(\\\"BLACK_FRAMES\\\")
\"
  '" 2>/dev/null || echo "ERROR")

  echo "[renderer-modes] Result for ${MODE}: ${_rgb_ok}"

  # Save server logs for this mode
  gcloud compute scp "${VM_NAME}:/tmp/geniesim_server.log" \
    "${LOCAL_RESULTS_DIR}/server_log_${MODE//+/_}.txt" \
    --zone="${VM_ZONE}" 2>/dev/null || true

  # Download episode JSONs
  gcloud compute scp --recurse \
    "${VM_NAME}:~/BlueprintPipeline/${SCENE_DIR}/geniesim/recordings/" \
    "${LOCAL_RESULTS_DIR}/recordings_${MODE//+/_}/" \
    --zone="${VM_ZONE}" 2>/dev/null || true

  # Clean recordings on VM for next mode test
  ssh_cmd bash -c "'
    rm -rf ~/BlueprintPipeline/${SCENE_DIR}/geniesim/recordings/*
  '" 2>/dev/null || true

  if [[ "${_rgb_ok}" == SUCCESS:* ]]; then
    _frames="${_rgb_ok#SUCCESS:}"
    echo ""
    echo "============================================================"
    echo " SUCCESS: Mode '${MODE}' produced ${_frames} RGB frames!"
    echo "============================================================"
    _success_mode="${MODE}"
    break
  fi

  echo "[renderer-modes] Mode '${MODE}' failed. Trying next mode..."
done

echo ""
echo "============================================================"
echo " Renderer Mode Sweep Results"
echo "============================================================"

if [ -n "${_success_mode}" ]; then
  echo "WINNER: ${_success_mode}"
  echo ""
  echo "To use this mode in production, set these env vars:"
  echo "  export BP_RENDERER_MODE=${_success_mode%%+*}"
  if [[ "${_success_mode}" == *"+RT2"* ]]; then
    echo "  export BP_RT2_ENABLED=1"
  fi
  echo ""
  echo "Then run the full test:"
  echo "  T4_AUTO_STOP=0 bash scripts/run-t4-rgb-test.sh"
  echo ""
  echo "Results saved to: ${LOCAL_RESULTS_DIR}"
else
  echo "ALL MODES FAILED. T4 cannot render RGB with any supported renderer."
  echo ""
  echo "Next step: Try L40S GPU on RunPod (Fallback 2):"
  echo "  bash scripts/setup-runpod-l40s.sh"
  echo ""
  echo "Logs for each mode saved to: ${LOCAL_RESULTS_DIR}"
fi
echo "============================================================"
