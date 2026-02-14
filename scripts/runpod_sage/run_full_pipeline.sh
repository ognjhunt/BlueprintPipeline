#!/usr/bin/env bash
# =============================================================================
# SAGE Full Pipeline Orchestrator (7 Stages + BlueprintPipeline Post-Processing)
#
# Runs ALL 7 SAGE stages end-to-end on a single pod:
#   1-3. Scene generation (robot-aware)
#   4.   Scene augmentation (MCP + offline)
#   5.   Grasp inference (M2T2)
#   6.   Trajectory planning (cuRobo / RRT)
#   7.   Data collection (HDF5 demos)
#   +    BP quality post-processing
#
# Usage:
#   ROOM_TYPE=kitchen \
#   ROBOT_TYPE=mobile_franka \
#   TASK_DESC="Pick up the mug from the counter and place it on the dining table" \
#   bash run_full_pipeline.sh
#
# Requirements:
#   - SAGE conda env activated
#   - SAM3D server running on :8080
#   - Isaac Sim running (optional, degrades gracefully)
#   - OPENAI_API_KEY set
# =============================================================================
set -euo pipefail

log() { echo "[pipeline $(date -u +%FT%TZ)] $*"; }

# ── Config ──────────────────────────────────────────────────────────────────
WORKSPACE="${WORKSPACE:-/workspace}"
SAGE_DIR="${WORKSPACE}/SAGE"
BP_DIR="${WORKSPACE}/BlueprintPipeline"
SAGE_SCRIPTS="${BP_DIR}/scripts/runpod_sage"

ROOM_TYPE="${ROOM_TYPE:-kitchen}"
ROBOT_TYPE="${ROBOT_TYPE:-mobile_franka}"
TASK_DESC="${TASK_DESC:-Pick up the mug from the counter and place it on the dining table}"
NUM_POSE_SAMPLES="${NUM_POSE_SAMPLES:-8}"
NUM_DEMOS="${NUM_DEMOS:-16}"
SKIP_AUGMENTATION="${SKIP_AUGMENTATION:-0}"
SKIP_GRASPS="${SKIP_GRASPS:-0}"
SKIP_DATA_GEN="${SKIP_DATA_GEN:-0}"
SKIP_BP_POSTPROCESS="${SKIP_BP_POSTPROCESS:-0}"
ENABLE_CAMERAS="${ENABLE_CAMERAS:-1}"
STRICT_PIPELINE="${STRICT_PIPELINE:-1}"
# NOTE: SAGE's physics MCP server starts Isaac Sim (noisy stdout). If enabled, ensure your MCP stdio stack
# is patched to ignore non-JSON stdout and Isaac Sim can run headless. Default off for robustness.
PHYSICS_CRITIC_ENABLED="${PHYSICS_CRITIC_ENABLED:-false}"

# Strict mode disallows skipping required stages.
if [[ "${STRICT_PIPELINE}" == "1" ]]; then
    if [[ "${SKIP_AUGMENTATION}" == "1" || "${SKIP_GRASPS}" == "1" || "${SKIP_DATA_GEN}" == "1" ]]; then
        log "ERROR: STRICT_PIPELINE=1 but one or more stages are skipped (SKIP_AUGMENTATION/SKIP_GRASPS/SKIP_DATA_GEN)."
        exit 2
    fi
fi

# Ensure conda is available
if ! command -v conda &>/dev/null; then
    source "${WORKSPACE}/miniconda3/etc/profile.d/conda.sh"
fi
conda activate sage 2>/dev/null || true

export SLURM_JOB_ID="${SLURM_JOB_ID:-12345}"

log "=========================================="
log "SAGE Full Pipeline (7 Stages + BP)"
log "=========================================="
log "Room:  ${ROOM_TYPE}"
log "Robot: ${ROBOT_TYPE}"
log "Task:  ${TASK_DESC}"
log "Demos: ${NUM_DEMOS}"
log ""

PIPELINE_START=$(date +%s)

# ── STAGES 1-3: Scene Generation (robot-aware) ─────────────────────────────
log "════ STAGES 1-3: Scene Generation ════"
cd "${SAGE_DIR}/client"

STAGE13_LOG="/tmp/sage_stage13_$(date +%s).log"
log "Running robot task client... (log: ${STAGE13_LOG})"

export PHYSICS_CRITIC_ENABLED
SERVER_PATHS=(../server/layout.py)
if [[ "${PHYSICS_CRITIC_ENABLED}" == "1" || "${PHYSICS_CRITIC_ENABLED}" == "true" ]]; then
    SERVER_PATHS+=(../server/physics/physics.py)
fi

python client_generation_robot_task.py \
    --room_type "${ROOM_TYPE}" \
    --robot_type "${ROBOT_TYPE}" \
    --task_description "${TASK_DESC}" \
    --server_paths "${SERVER_PATHS[@]}" \
    2>&1 | tee "${STAGE13_LOG}"

# Find the latest layout directory
LAYOUT_ID=$(ls -td "${SAGE_DIR}/server/results/layout_"* 2>/dev/null | head -1 | xargs basename)
if [[ -z "${LAYOUT_ID}" ]]; then
    log "ERROR: No layout directory found after scene generation"
    exit 1
fi
LAYOUT_DIR="${SAGE_DIR}/server/results/${LAYOUT_ID}"
log "Layout: ${LAYOUT_ID}"
log "Directory: ${LAYOUT_DIR}"

# Count generated objects
NUM_OBJECTS=$(ls "${LAYOUT_DIR}/generation/"*.obj 2>/dev/null | wc -l || echo "0")
log "Objects generated: ${NUM_OBJECTS}"

STAGE13_END=$(date +%s)
log "Stages 1-3 completed in $(( STAGE13_END - PIPELINE_START ))s"

# ── Unload SAM3D (free ~13GB VRAM for stages 5-7) ──────────────────────────
log ""
log "Unloading SAM3D to free VRAM..."
curl -sf -X POST http://localhost:8080/shutdown 2>/dev/null || \
    pkill -f sam3d_server.py 2>/dev/null || true
sleep 3
log "SAM3D unloaded"

# ── STAGE 4: Scene Augmentation ─────────────────────────────────────────────
if [[ "${SKIP_AUGMENTATION}" != "1" ]]; then
    log ""
    log "════ STAGE 4: Scene Augmentation ════"

    # 4a: MCP-based augmentation (agent adds complementary objects)
    log "4a: MCP-based augmentation..."
    AUG_CLIENT="${SAGE_DIR}/client/client_generation_scene_aug.py"
    if [[ -f "${AUG_CLIENT}" ]]; then
        if ! python "${AUG_CLIENT}" \
            --base_layout_dict_path "${LAYOUT_DIR}/${LAYOUT_ID}.json" \
            --server_paths "${SAGE_DIR}/server/layout.py" \
            --from_task_required_objects \
            2>&1 | tee "/tmp/sage_stage4a.log"; then
            if [[ "${STRICT_PIPELINE}" == "1" ]]; then
                log "ERROR: MCP augmentation failed in strict mode."
                exit 1
            fi
            log "WARNING: MCP augmentation failed (non-fatal)"
        fi
    else
        if [[ "${STRICT_PIPELINE}" == "1" ]]; then
            log "ERROR: Strict mode requires MCP augmentation script; missing ${AUG_CLIENT}"
            exit 1
        fi
        log "SKIP: client_generation_scene_aug.py not found"
    fi

    # 4b: Pose augmentation
    log "4b: Pose augmentation (${NUM_POSE_SAMPLES} samples)..."
    cd "${SAGE_DIR}/server"
    POSE_AUG_SCRIPT="augment/pose_aug_mm_from_layout_with_task.py"
    if [[ -f "${POSE_AUG_SCRIPT}" ]]; then
        if ! python "${POSE_AUG_SCRIPT}" \
            --layout_id "${LAYOUT_ID}" \
            --save_dir_name pose_aug_0 \
            --num_samples "${NUM_POSE_SAMPLES}" \
            2>&1 | tee "/tmp/sage_stage4b.log"; then
            if [[ "${STRICT_PIPELINE}" == "1" ]]; then
                log "ERROR: Pose augmentation failed in strict mode."
                exit 1
            fi
            log "WARNING: Pose augmentation failed (non-fatal)"
        fi
    else
        if [[ "${STRICT_PIPELINE}" == "1" ]]; then
            log "ERROR: Strict mode requires pose augmentation script; missing ${POSE_AUG_SCRIPT}"
            exit 1
        fi
        log "SKIP: pose augmentation script not found"
    fi

    # Stage 4 MUST produce pose_aug meta with at least 1 layout.
    POSE_META="${LAYOUT_DIR}/pose_aug_0/meta.json"
    if [[ ! -f "${POSE_META}" ]]; then
        log "ERROR: Stage 4 failed — missing ${POSE_META}"
        exit 1
    fi
    NUM_LAYOUTS_IN_META="$(python3 - <<PY
import json, sys
path = "${POSE_META}"
data = json.load(open(path, "r"))
count = 0
if isinstance(data, list):
    count = len(data)
elif isinstance(data, dict):
    for key in ("layouts", "layout_dict_paths", "layout_paths", "variants", "feasible_layouts"):
        if key in data and isinstance(data[key], list):
            count = len(data[key])
            break
print(count)
PY
)"
    if [[ "${NUM_LAYOUTS_IN_META}" -lt 1 ]]; then
        log "ERROR: Stage 4 failed — ${POSE_META} contains 0 layouts"
        exit 1
    fi
    log "Stage 4 meta OK: ${NUM_LAYOUTS_IN_META} pose-aug layouts"

    STAGE4_END=$(date +%s)
    log "Stage 4 completed in $(( STAGE4_END - STAGE13_END ))s"
else
    log ""
    log "════ STAGE 4: SKIPPED (SKIP_AUGMENTATION=1) ════"
    STAGE4_END=$(date +%s)
fi

# ── STAGES 5-7: Grasp + Planning + Data Collection (STRICT) ────────────────
if [[ "${SKIP_GRASPS}" == "1" || "${SKIP_DATA_GEN}" == "1" ]]; then
    log ""
    log "════ STAGES 5-7: SKIPPED (SKIP_GRASPS=1 or SKIP_DATA_GEN=1) ════"
    STAGE67_END=$(date +%s)
else
    log ""
    log "════ STAGES 5-7: Grasp + Planning + Data Collection ════"

    # If Isaac Sim MCP is running (started by entrypoint), stop it before launching
    # the Stage 7 collector which starts its own headless SimulationApp.
    if [[ -f /tmp/isaacsim.pid ]]; then
        ISAAC_PID="$(cat /tmp/isaacsim.pid || true)"
        if [[ -n "${ISAAC_PID}" ]] && kill -0 "${ISAAC_PID}" 2>/dev/null; then
            log "Stopping Isaac Sim MCP service (pid=${ISAAC_PID}) to free VRAM..."
            kill "${ISAAC_PID}" 2>/dev/null || true
            for _ in $(seq 1 60); do
                if ! kill -0 "${ISAAC_PID}" 2>/dev/null; then
                    break
                fi
                sleep 2
            done
            if kill -0 "${ISAAC_PID}" 2>/dev/null; then
                log "Isaac Sim still alive — sending SIGKILL"
                kill -9 "${ISAAC_PID}" 2>/dev/null || true
            fi
        fi
    fi

    STAGE567_SCRIPT="${SAGE_SCRIPTS}/sage_stage567_mobile_franka.py"
    if [[ ! -f "${STAGE567_SCRIPT}" ]]; then
        log "ERROR: Missing ${STAGE567_SCRIPT}"
        exit 1
    fi

    STAGE567_ARGS=(--layout_id "${LAYOUT_ID}" --results_dir "${SAGE_DIR}/server/results" --pose_aug_name pose_aug_0 --num_demos "${NUM_DEMOS}")
    [[ "${ENABLE_CAMERAS}" == "1" ]] && STAGE567_ARGS+=(--enable_cameras)
    STAGE567_ARGS+=(--headless)
    [[ "${STRICT_PIPELINE}" == "1" ]] && STAGE567_ARGS+=(--strict)

    python "${STAGE567_SCRIPT}" "${STAGE567_ARGS[@]}" 2>&1 | tee "/tmp/sage_stage567.log"

    # Required artifacts
    if [[ ! -f "${LAYOUT_DIR}/grasps/grasp_transforms.json" ]]; then
        log "ERROR: Missing grasps output: ${LAYOUT_DIR}/grasps/grasp_transforms.json"
        exit 1
    fi
    if [[ ! -f "${LAYOUT_DIR}/demos/dataset.hdf5" ]]; then
        log "ERROR: Missing demos output: ${LAYOUT_DIR}/demos/dataset.hdf5"
        exit 1
    fi

    STAGE67_END=$(date +%s)
    log "Stages 5-7 completed in $(( STAGE67_END - STAGE4_END ))s"
fi

# ── BlueprintPipeline Post-Processing ──────────────────────────────────────
if [[ "${SKIP_BP_POSTPROCESS}" != "1" ]]; then
    log ""
    log "════ BP POST-PROCESSING ════"

    BP_POSTPROCESS="${SAGE_SCRIPTS}/sage_to_bp_postprocess.py"
    if [[ -f "${BP_POSTPROCESS}" ]]; then
        OUTPUT_DIR="${WORKSPACE}/outputs/${LAYOUT_ID}_bp"
        python "${BP_POSTPROCESS}" \
            --sage_results "${LAYOUT_DIR}" \
            --output_dir "${OUTPUT_DIR}" \
            2>&1 | tee "/tmp/sage_bp_postprocess.log"
        log "BP output: ${OUTPUT_DIR}"

        if [[ ! -f "${OUTPUT_DIR}/quality_report.json" ]]; then
            log "ERROR: BP post-processing did not produce ${OUTPUT_DIR}/quality_report.json"
            exit 1
        fi
    else
        log "SKIP: sage_to_bp_postprocess.py not found"
    fi

    BP_END=$(date +%s)
    log "BP post-processing completed in $(( BP_END - STAGE67_END ))s"
else
    log ""
    log "════ BP POST-PROCESSING: SKIPPED ════"
    BP_END=$(date +%s)
fi

# ── Summary ────────────────────────────────────────────────────────────────
PIPELINE_END=$(date +%s)
TOTAL_TIME=$(( PIPELINE_END - PIPELINE_START ))

log ""
log "=========================================="
log "Pipeline Complete!"
log "=========================================="
log "Layout:     ${LAYOUT_ID}"
log "Objects:    ${NUM_OBJECTS}"
log "Total time: ${TOTAL_TIME}s ($(( TOTAL_TIME / 60 ))m $(( TOTAL_TIME % 60 ))s)"
log ""
log "Outputs:"
log "  Scene:    ${LAYOUT_DIR}/"
log "  Grasps:   ${LAYOUT_DIR}/grasps/"
log "  Demos:    ${LAYOUT_DIR}/demos/"
[[ "${SKIP_BP_POSTPROCESS}" != "1" ]] && log "  Quality:  ${WORKSPACE}/outputs/${LAYOUT_ID}_bp/"
log ""
log "Logs:"
log "  Stages 1-3: ${STAGE13_LOG}"
log "  Stage 4:    /tmp/sage_stage4a.log, /tmp/sage_stage4b.log"
log "  Stages 5-7: /tmp/sage_stage567.log"
log "  BP:         /tmp/sage_bp_postprocess.log"
log "=========================================="
