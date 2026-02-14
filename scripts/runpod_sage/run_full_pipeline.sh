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

python client_generation_robot_task.py \
    --room_type "${ROOM_TYPE}" \
    --robot_type "${ROBOT_TYPE}" \
    --task_description "${TASK_DESC}" \
    --server_paths ../server/layout.py ../server/physics/physics.py \
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
        python "${AUG_CLIENT}" \
            --base_layout_dict_path "${LAYOUT_DIR}/${LAYOUT_ID}.json" \
            --server_paths "${SAGE_DIR}/server/layout.py" \
            --from_task_required_objects \
            2>&1 | tee "/tmp/sage_stage4a.log" || \
            log "WARNING: MCP augmentation failed (non-fatal)"
    else
        log "SKIP: client_generation_scene_aug.py not found"
    fi

    # 4b: Pose augmentation
    log "4b: Pose augmentation (${NUM_POSE_SAMPLES} samples)..."
    cd "${SAGE_DIR}/server"
    POSE_AUG_SCRIPT="augment/pose_aug_mm_from_layout_with_task.py"
    if [[ -f "${POSE_AUG_SCRIPT}" ]]; then
        python "${POSE_AUG_SCRIPT}" \
            --layout_id "${LAYOUT_ID}" \
            --save_dir_name pose_aug_0 \
            --num_samples "${NUM_POSE_SAMPLES}" \
            2>&1 | tee "/tmp/sage_stage4b.log" || \
            log "WARNING: Pose augmentation failed (non-fatal)"
    else
        log "SKIP: pose augmentation script not found"
    fi

    STAGE4_END=$(date +%s)
    log "Stage 4 completed in $(( STAGE4_END - STAGE13_END ))s"
else
    log ""
    log "════ STAGE 4: SKIPPED (SKIP_AUGMENTATION=1) ════"
    STAGE4_END=$(date +%s)
fi

# ── STAGE 5: Grasp Inference (M2T2) ────────────────────────────────────────
if [[ "${SKIP_GRASPS}" != "1" ]]; then
    log ""
    log "════ STAGE 5: Grasp Inference (M2T2) ════"
    cd "${SAGE_DIR}/server"

    GRASP_SCRIPT="${SAGE_SCRIPTS}/render_and_infer_grasps.py"
    if [[ -f "${GRASP_SCRIPT}" ]]; then
        python "${GRASP_SCRIPT}" \
            --layout_id "${LAYOUT_ID}" \
            --results_dir "${SAGE_DIR}/server/results" \
            --num_views 4 \
            2>&1 | tee "/tmp/sage_stage5.log" || \
            log "WARNING: Grasp inference failed (non-fatal)"
    else
        log "SKIP: render_and_infer_grasps.py not found"
    fi

    STAGE5_END=$(date +%s)
    log "Stage 5 completed in $(( STAGE5_END - STAGE4_END ))s"
else
    log ""
    log "════ STAGE 5: SKIPPED (SKIP_GRASPS=1) ════"
    STAGE5_END=$(date +%s)
fi

# ── STAGES 6-7: Trajectory Planning + Data Collection ──────────────────────
if [[ "${SKIP_DATA_GEN}" != "1" ]]; then
    log ""
    log "════ STAGES 6-7: Trajectory Planning + Data Collection ════"

    DATAGEN_SCRIPT="${SAGE_SCRIPTS}/data_generation_mobile_franka.py"
    if [[ -f "${DATAGEN_SCRIPT}" ]]; then
        DATAGEN_ARGS="--layout_id ${LAYOUT_ID} --results_dir ${SAGE_DIR}/server/results --num_demos ${NUM_DEMOS}"
        [[ "${ENABLE_CAMERAS}" == "1" ]] && DATAGEN_ARGS="${DATAGEN_ARGS} --enable_cameras"
        DATAGEN_ARGS="${DATAGEN_ARGS} --headless"

        python "${DATAGEN_SCRIPT}" ${DATAGEN_ARGS} \
            2>&1 | tee "/tmp/sage_stage67.log" || \
            log "WARNING: Data generation failed (non-fatal)"
    else
        log "SKIP: data_generation_mobile_franka.py not found"
    fi

    STAGE67_END=$(date +%s)
    log "Stages 6-7 completed in $(( STAGE67_END - STAGE5_END ))s"
else
    log ""
    log "════ STAGES 6-7: SKIPPED (SKIP_DATA_GEN=1) ════"
    STAGE67_END=$(date +%s)
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
            2>&1 | tee "/tmp/sage_bp_postprocess.log" || \
            log "WARNING: BP post-processing failed (non-fatal)"
        log "BP output: ${OUTPUT_DIR}"
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
log "  Stage 5:    /tmp/sage_stage5.log"
log "  Stages 6-7: /tmp/sage_stage67.log"
log "  BP:         /tmp/sage_bp_postprocess.log"
log "=========================================="
