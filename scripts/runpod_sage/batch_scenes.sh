#!/usr/bin/env bash
# =============================================================================
# Batch Scene Runner — runs multiple SAGE scenes sequentially with logging.
#
# Usage:
#   bash batch_scenes.sh
#
# Override defaults via env vars:
#   NUM_DEMOS=8 NUM_POSE_SAMPLES=4 bash batch_scenes.sh
# =============================================================================
set -euo pipefail

log() { echo "[batch $(date -u +%FT%TZ)] $*"; }

WORKSPACE="${WORKSPACE:-/workspace}"
SAGE_SCRIPTS="${WORKSPACE}/BlueprintPipeline/scripts/runpod_sage"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKSPACE}/outputs}"
LOGDIR="${OUTPUT_ROOT}/logs"
mkdir -p "${LOGDIR}"

NUM_DEMOS="${NUM_DEMOS:-16}"
NUM_POSE_SAMPLES="${NUM_POSE_SAMPLES:-8}"
ROBOT_TYPE="${ROBOT_TYPE:-mobile_franka}"
ENABLE_CAMERAS="${ENABLE_CAMERAS:-1}"

# ── Scene Definitions ─────────────────────────────────────────────────────
# Format: room_type|task_description
SCENES=(
  "kitchen|Pick up the mug from the counter and place it on the dining table"
  "office|Sort the papers on the desk and place them in the organizer"
  "warehouse|Pick up the small box from the shelf and place it in the shipping bin"
  "living_room|Pick up the remote from the coffee table and place it on the sofa armrest"
  "lab|Pick up the beaker from the lab bench and place it on the drying rack"
)

# ── Main Loop ─────────────────────────────────────────────────────────────
TOTAL=${#SCENES[@]}
PASSED=0
FAILED=0
FAILED_SCENES=""

log "Starting batch: ${TOTAL} scenes, robot=${ROBOT_TYPE}, demos=${NUM_DEMOS}, pose_samples=${NUM_POSE_SAMPLES}"

for i in "${!SCENES[@]}"; do
  IFS='|' read -r ROOM_TYPE TASK_DESC <<< "${SCENES[$i]}"
  RUN_NUM=$((i + 1))
  LOGFILE="${LOGDIR}/scene_${ROOM_TYPE}_$(date +%s).log"

  log "━━━ Scene ${RUN_NUM}/${TOTAL}: ${ROOM_TYPE} ━━━"
  log "  Task: ${TASK_DESC}"
  log "  Log: ${LOGFILE}"

  if ROOM_TYPE="${ROOM_TYPE}" \
     ROBOT_TYPE="${ROBOT_TYPE}" \
     TASK_DESC="${TASK_DESC}" \
     NUM_DEMOS="${NUM_DEMOS}" \
     NUM_POSE_SAMPLES="${NUM_POSE_SAMPLES}" \
     ENABLE_CAMERAS="${ENABLE_CAMERAS}" \
     STRICT_PIPELINE=1 \
     bash "${SAGE_SCRIPTS}/run_full_pipeline.sh" > "${LOGFILE}" 2>&1; then
    PASSED=$((PASSED + 1))
    log "  ✓ ${ROOM_TYPE} completed"
  else
    FAILED=$((FAILED + 1))
    FAILED_SCENES="${FAILED_SCENES} ${ROOM_TYPE}"
    log "  ✗ ${ROOM_TYPE} FAILED (see ${LOGFILE})"
  fi

  # Check disk space between runs
  AVAIL=$(df --output=avail /workspace 2>/dev/null | tail -1 | tr -d ' ')
  if [[ -n "${AVAIL}" ]] && [[ "${AVAIL}" -lt 10485760 ]]; then
    log "WARNING: <10GB disk remaining. Stopping batch."
    break
  fi
done

# ── Summary ───────────────────────────────────────────────────────────────
log "━━━ Batch Complete ━━━"
log "  Total: ${TOTAL}  Passed: ${PASSED}  Failed: ${FAILED}"
if [[ ${FAILED} -gt 0 ]]; then
  log "  Failed scenes:${FAILED_SCENES}"
fi
log "  Logs: ${LOGDIR}"
log "  Outputs: ${OUTPUT_ROOT}"
