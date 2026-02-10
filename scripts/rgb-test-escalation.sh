#!/usr/bin/env bash
set -euo pipefail
# =============================================================================
# Unified RGB Test Orchestrator with Automatic Fallback Escalation
#
# This script automates the full escalation ladder:
#   1. T4 with RaytracedLighting (default)
#   2. T4 with alternative renderer modes (PathTracing, RT2)
#   3. L40S on RunPod
#
# It stops as soon as any level succeeds.
#
# Usage:
#   bash scripts/rgb-test-escalation.sh
#
# To skip T4 and go straight to renderer modes:
#   SKIP_T4_DEFAULT=1 bash scripts/rgb-test-escalation.sh
#
# To skip directly to L40S:
#   SKIP_T4=1 bash scripts/rgb-test-escalation.sh
#
# Cost estimate: $1.50 (T4 only) → $3-5 (if L40S needed)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"

SKIP_T4_DEFAULT="${SKIP_T4_DEFAULT:-0}"
SKIP_T4="${SKIP_T4:-0}"
RESULTS_BASE="${RGB_RESULTS_DIR:-./rgb_escalation_results}"

_timestamp=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${RESULTS_BASE}/${_timestamp}"
mkdir -p "${RESULTS_DIR}"

# Log everything
exec > >(tee -a "${RESULTS_DIR}/escalation.log") 2>&1

echo "============================================================"
echo " RGB Test Escalation Orchestrator"
echo " $(date)"
echo "============================================================"
echo "Results: ${RESULTS_DIR}"
echo ""

_success=false
_winning_level=""
_winning_details=""

# ── Helper: Check if T4 VM exists ────────────────────────────────────────────
t4_vm_exists() {
  local vm_name="${T4_VM_NAME:-geniesim-t4-test}"
  local vm_zone="${T4_VM_ZONE:-us-east1-b}"
  gcloud compute instances describe "${vm_name}" --zone="${vm_zone}" &>/dev/null
}

# ── Helper: Analyze test results ─────────────────────────────────────────────
check_rgb_success() {
  local results_dir="$1"
  # Look for any episode JSON with camera_capture_frames > 0
  python3 -c "
import json, glob, os, sys
json_files = sorted(glob.glob(os.path.join('${results_dir}', '**', '*.json'), recursive=True))
total_camera_frames = 0
for f in json_files:
    try:
        with open(f) as fh:
            data = json.load(fh)
        meta = data.get('metadata', {})
        quality = meta.get('quality_breakdown', {})
        camera = quality.get('camera_frames', {})
        captured = camera.get('camera_capture_frames', 0)
        total_camera_frames += captured
    except Exception:
        pass
sys.exit(0 if total_camera_frames > 0 else 1)
" 2>/dev/null
}

# =============================================================================
# Level 1: T4 with Default Renderer (RaytracedLighting)
# =============================================================================
if [ "${SKIP_T4}" = "0" ] && [ "${SKIP_T4_DEFAULT}" = "0" ]; then
  echo ""
  echo "============================================================"
  echo " LEVEL 1: T4 + RaytracedLighting (default)"
  echo "============================================================"

  if ! t4_vm_exists; then
    echo "[escalation] T4 VM doesn't exist. Creating..."
    bash scripts/create-t4-test-vm.sh 2>&1 | tee "${RESULTS_DIR}/create_t4.log"
    echo "[escalation] Setting up T4 VM..."
    bash scripts/setup-t4-vm.sh 2>&1 | tee "${RESULTS_DIR}/setup_t4.log"
  fi

  echo "[escalation] Running T4 RGB test..."
  T4_AUTO_STOP=0 T4_RESULTS_DIR="${RESULTS_DIR}/t4_default" \
    bash scripts/run-t4-rgb-test.sh 2>&1 | tee "${RESULTS_DIR}/t4_default_test.log" || true

  if check_rgb_success "${RESULTS_DIR}/t4_default"; then
    _success=true
    _winning_level="Level 1"
    _winning_details="T4 + RaytracedLighting"
  else
    echo "[escalation] Level 1 FAILED. RGB frames = 0."
  fi
fi

# =============================================================================
# Level 2: T4 with Alternative Renderer Modes
# =============================================================================
if [ "${_success}" = "false" ] && [ "${SKIP_T4}" = "0" ]; then
  echo ""
  echo "============================================================"
  echo " LEVEL 2: T4 + Alternative Renderer Modes"
  echo "============================================================"

  T4_RESULTS_DIR="${RESULTS_DIR}/t4_renderer_modes" \
    bash scripts/try-renderer-modes.sh 2>&1 | tee "${RESULTS_DIR}/renderer_modes.log" || true

  if check_rgb_success "${RESULTS_DIR}/t4_renderer_modes"; then
    _success=true
    _winning_level="Level 2"
    # Extract winning mode from log
    _winning_details="T4 + $(grep 'WINNER:' "${RESULTS_DIR}/renderer_modes.log" | head -1 | sed 's/.*WINNER: //' || echo 'alt renderer')"
  else
    echo "[escalation] Level 2 FAILED. All renderer modes produce black frames on T4."
  fi

  # Stop T4 VM to save money before trying L40S
  echo "[escalation] Stopping T4 VM..."
  T4_VM_NAME="${T4_VM_NAME:-geniesim-t4-test}"
  T4_VM_ZONE="${T4_VM_ZONE:-us-east1-b}"
  gcloud compute instances stop "${T4_VM_NAME}" --zone="${T4_VM_ZONE}" 2>/dev/null || true
fi

# =============================================================================
# Level 3: L40S on RunPod
# =============================================================================
if [ "${_success}" = "false" ]; then
  echo ""
  echo "============================================================"
  echo " LEVEL 3: L40S on RunPod"
  echo "============================================================"
  echo ""
  echo "T4 has failed (all renderer modes). L40S is the next step."
  echo ""

  # Check if RunPod is configured
  if command -v runpodctl &>/dev/null; then
    echo "[escalation] RunPod CLI available. Launching L40S setup..."
    L40S_RESULTS_DIR="${RESULTS_DIR}/l40s" \
      bash scripts/setup-runpod-l40s.sh 2>&1 | tee "${RESULTS_DIR}/l40s_setup.log" || true

    echo ""
    echo "[escalation] L40S pod created. You need to:"
    echo "  1. Get SSH details from RunPod dashboard"
    echo "  2. Run: bash scripts/run-runpod-rgb-test.sh <host> <port>"
    echo ""
    echo "  After test completes, results will be in: ${RESULTS_DIR}/l40s/"
  else
    echo "[escalation] RunPod CLI not installed."
    echo ""
    echo "To set up L40S manually:"
    echo "  1. pip install runpodctl"
    echo "  2. export RUNPOD_API_KEY=<your-api-key>"
    echo "  3. bash scripts/setup-runpod-l40s.sh"
    echo ""
    echo "Or rent an L40S from:"
    echo "  - RunPod: https://www.runpod.io (~\$1.50/hr)"
    echo "  - Lambda: https://cloud.lambdalabs.com (~\$1.50/hr)"
    echo "  - Vast.ai: https://vast.ai (~\$0.80/hr spot)"
    echo ""
    echo "Then run: bash scripts/run-runpod-rgb-test.sh <host> <port>"
  fi
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo " RGB Test Escalation — Summary"
echo "============================================================"

if [ "${_success}" = "true" ]; then
  echo " RESULT: SUCCESS"
  echo " Level:  ${_winning_level}"
  echo " Config: ${_winning_details}"
  echo ""
  echo " RGB rendering is working! Next steps:"
  echo "   1. Run the full pipeline with all 7 tasks using this config"
  echo "   2. Download and validate the episode data"
  echo "   3. Update production config if needed"
else
  echo " RESULT: T4 EXHAUSTED — L40S NEEDED"
  echo ""
  echo " All T4 renderer modes have been tried and failed."
  echo " An L40S pod has been set up (or needs to be set up)."
  echo " Follow the L40S instructions above to complete the test."
fi

echo ""
echo " All logs saved to: ${RESULTS_DIR}"
echo "============================================================"
