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
#   +    Interactive backends (PhysX-Anything, Infinigen) [opt: ENABLE_INTERACTIVE=1]
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

_norm_room_type() {
    local rt="${1:-}"
    rt="${rt,,}"
    rt="${rt// /_}"
    rt="${rt//-/_}"
    echo "${rt}"
}

_default_max_objects_for_room_type() {
    # Conservative defaults for smoke tests and end-to-end validation.
    # Override explicitly with `SAGE_MAX_OBJECTS=<int>` (set to 0 to disable).
    local rt="$(_norm_room_type "${1:-}")"
    case "${rt}" in
        *house*|*multi*|*open_plan*|*floorplan*|*home*) echo 45 ;;
        *apartment*|*studio*) echo 35 ;;
        *kitchen*) echo 28 ;;
        *living*|*family*|*lounge*) echo 25 ;;
        *dining*) echo 24 ;;
        *bed*) echo 22 ;;
        *office*|*study*) echo 22 ;;
        *garage*|*workshop*) echo 25 ;;
        *bath*|*restroom*) echo 16 ;;
        *hall*|*corridor*|*entry*|*foyer*) echo 15 ;;
        *closet*|*pantry*|*laundry*) echo 18 ;;
        *) echo 24 ;;
    esac
}

# ── Config ──────────────────────────────────────────────────────────────────
WORKSPACE="${WORKSPACE:-/workspace}"
SAGE_DIR="${WORKSPACE}/SAGE"
BP_DIR="${WORKSPACE}/BlueprintPipeline"
SAGE_SCRIPTS="${BP_DIR}/scripts/runpod_sage"

SCENE_SOURCE="${SCENE_SOURCE:-sage}"  # sage|scenesmith

ROOM_TYPE="${ROOM_TYPE:-kitchen}"
ROBOT_TYPE="${ROBOT_TYPE:-mobile_franka}"
TASK_DESC="${TASK_DESC:-Pick up the mug from the counter and place it on the dining table}"
NUM_POSE_SAMPLES="${NUM_POSE_SAMPLES:-8}"
NUM_DEMOS="${NUM_DEMOS:-16}"
SKIP_AUGMENTATION="${SKIP_AUGMENTATION:-0}"
SKIP_GRASPS="${SKIP_GRASPS:-0}"
SKIP_DATA_GEN="${SKIP_DATA_GEN:-0}"
SKIP_BP_POSTPROCESS="${SKIP_BP_POSTPROCESS:-0}"
ENABLE_INTERACTIVE="${ENABLE_INTERACTIVE:-0}"
ENABLE_CAMERAS="${ENABLE_CAMERAS:-1}"
STRICT_PIPELINE="${STRICT_PIPELINE:-1}"
# Scene quality gates (SceneSmith-style repairs; CPU-only).
SAGE_QUALITY_PROFILE="${SAGE_QUALITY_PROFILE:-standard}"  # standard|strict
SAGE_QUALITY_FALLBACK_ENABLED="${SAGE_QUALITY_FALLBACK_ENABLED:-1}"  # 1|0
SAGE_QUALITY_MAX_ITERS="${SAGE_QUALITY_MAX_ITERS:-6}"
POSE_AUG_NAME="${POSE_AUG_NAME:-pose_aug_0}"
# NOTE: SAGE's physics MCP server starts Isaac Sim (noisy stdout). If enabled, ensure your MCP stdio stack
# is patched to ignore non-JSON stdout and Isaac Sim can run headless. Default off for robustness.
PHYSICS_CRITIC_ENABLED="${PHYSICS_CRITIC_ENABLED:-false}"

# Object cap (prevents runaway clutter that explodes SAM3D time).
# If unset, choose a default by room type; set `SAGE_MAX_OBJECTS=0` to disable.
if [[ -z "${SAGE_MAX_OBJECTS+x}" ]]; then
    SAGE_MAX_OBJECTS="$(_default_max_objects_for_room_type "${ROOM_TYPE}")"
fi
if [[ ! "${SAGE_MAX_OBJECTS}" =~ ^[0-9]+$ ]]; then
    log "WARNING: SAGE_MAX_OBJECTS must be an integer; got '${SAGE_MAX_OBJECTS}'. Disabling cap."
    SAGE_MAX_OBJECTS=0
fi
export SAGE_MAX_OBJECTS

TASK_DESC_CAPPED="${TASK_DESC}"
if [[ "${SAGE_MAX_OBJECTS}" -gt 0 ]]; then
    TASK_DESC_CAPPED="${TASK_DESC}. Scene constraints: Keep total object count <= ${SAGE_MAX_OBJECTS} (including fixtures, furniture, appliances, and small items). Never omit task-referenced objects. Prefer essential surfaces + task-critical items; omit low-importance decor/clutter once you hit the cap."
fi

# Strict mode disallows skipping required stages.
if [[ "${STRICT_PIPELINE}" == "1" ]]; then
    if [[ "${SCENE_SOURCE}" == "sage" ]]; then
        if [[ "${SKIP_AUGMENTATION}" == "1" || "${SKIP_GRASPS}" == "1" || "${SKIP_DATA_GEN}" == "1" ]]; then
            log "ERROR: STRICT_PIPELINE=1 but one or more stages are skipped (SKIP_AUGMENTATION/SKIP_GRASPS/SKIP_DATA_GEN)."
            exit 2
        fi
    else
        # SceneSmith mode provides its own pose_aug meta; Stage 4 augmentation is intentionally skipped.
        if [[ "${SKIP_GRASPS}" == "1" || "${SKIP_DATA_GEN}" == "1" ]]; then
            log "ERROR: STRICT_PIPELINE=1 but one or more stages are skipped (SKIP_GRASPS/SKIP_DATA_GEN)."
            exit 2
        fi
    fi
fi

# Ensure conda is available
if ! command -v conda &>/dev/null; then
    source "${WORKSPACE}/miniconda3/etc/profile.d/conda.sh"
fi
conda activate sage 2>/dev/null || true

export SLURM_JOB_ID="${SLURM_JOB_ID:-12345}"

# Prefer system Python for helper scripts (requests/trimesh live outside conda in the image).
PY_SYS="${PY_SYS:-python3.11}"
if ! command -v "${PY_SYS}" &>/dev/null; then
    PY_SYS="python3"
fi

log "=========================================="
log "SAGE Full Pipeline (7 Stages + BP + Interactive)"
log "=========================================="
log "Scene source: ${SCENE_SOURCE}"
log "Quality: ${SAGE_QUALITY_PROFILE} (fallback=${SAGE_QUALITY_FALLBACK_ENABLED}, iters=${SAGE_QUALITY_MAX_ITERS})"
log "Interactive: ${ENABLE_INTERACTIVE}"
log "Room:  ${ROOM_TYPE}"
log "Robot: ${ROBOT_TYPE}"
log "Task:  ${TASK_DESC}"
log "Max objects: ${SAGE_MAX_OBJECTS} (override with SAGE_MAX_OBJECTS=..., set 0 to disable)"
log "Demos: ${NUM_DEMOS}"
log ""

PIPELINE_START=$(date +%s)

# ── SAM3D health check helper ─────────────────────────────────────────────
SAM3D_PORT="${SAM3D_PORT:-8080}"
SAM3D_URL="http://127.0.0.1:${SAM3D_PORT}"
SCENE_GEN_TIMEOUT="${SCENE_GEN_TIMEOUT:-7200}"  # 2 hours max for scene generation (meshes can take ~1-2 min/object)

_ensure_sam3d_healthy() {
    local status
    status=$(curl -sf -o /dev/null -w "%{http_code}" "${SAM3D_URL}/health" 2>/dev/null || echo "000")
    if [[ "${status}" == "200" ]]; then
        return 0
    fi
    log "SAM3D health check failed (status=${status}). Restarting..."

    pkill -f sam3d_server.py 2>/dev/null || true
    sleep 3

    # Determine python binary for SAM3D
    local sam3d_py="${PY_SYS}"
    [[ -x "/workspace/miniconda3/envs/sage/bin/python" ]] && sam3d_py="/workspace/miniconda3/envs/sage/bin/python"

    local sam3d_args="--port ${SAM3D_PORT} --image-backend ${SAM3D_IMAGE_BACKEND:-gemini}"
    [[ -f /workspace/sam3d/checkpoints/hf/pipeline.yaml ]] && sam3d_args="${sam3d_args} --checkpoint-dir /workspace/sam3d/checkpoints/hf"

    nohup "${sam3d_py}" "${SAGE_SCRIPTS}/sam3d_server.py" ${sam3d_args} >> /tmp/sam3d_server.log 2>&1 &
    local new_pid=$!
    echo "${new_pid}" > /tmp/sam3d.pid
    log "SAM3D restarted (PID=${new_pid}). Waiting for health..."

    local deadline=$(( $(date +%s) + 300 ))
    while [[ "$(date +%s)" -lt "${deadline}" ]]; do
        if curl -sf "${SAM3D_URL}/health" >/dev/null 2>&1; then
            log "SAM3D healthy after restart."
            return 0
        fi
        if ! kill -0 "${new_pid}" 2>/dev/null; then
            log "ERROR: SAM3D died during restart."
            return 1
        fi
        sleep 3
    done
    log "ERROR: SAM3D did not become healthy within 300s."
    return 1
}

# ── SCENE GENERATION ───────────────────────────────────────────────────────
if [[ "${SCENE_SOURCE}" == "scenesmith" ]]; then
    log "════ SCENE GENERATION: SceneSmith → SAGE Layout Dir ════"
    STAGE13_LOG="/tmp/scenesmith_stage1_$(date +%s).log"
    log "Running SceneSmith paper stack bridge... (log: ${STAGE13_LOG})"

    SCENESMITH_TO_SAGE="${SAGE_SCRIPTS}/scenesmith_to_sage_layout.py"
    if [[ ! -f "${SCENESMITH_TO_SAGE}" ]]; then
        log "ERROR: Missing ${SCENESMITH_TO_SAGE}"
        exit 1
    fi

    # SceneSmith mode intentionally skips SAGE Stage 4 augmentation.
    SKIP_AUGMENTATION=1

    LAYOUT_ID="$("${PY_SYS}" "${SCENESMITH_TO_SAGE}" \
        --results_dir "${SAGE_DIR}/server/results" \
        --room_type "${ROOM_TYPE}" \
        --task_desc "${TASK_DESC_CAPPED}" \
        --pose_aug_name "${POSE_AUG_NAME}" \
        2> >(tee "${STAGE13_LOG}" >&2))"
    echo "${LAYOUT_ID}" >> "${STAGE13_LOG}"
    if [[ -z "${LAYOUT_ID}" ]]; then
        log "ERROR: SceneSmith->SAGE bridge did not return a layout_id"
        exit 1
    fi
    LAYOUT_DIR="${SAGE_DIR}/server/results/${LAYOUT_ID}"
else
    log "════ STAGES 1-3: Scene Generation (SAGE) ════"

    # Pre-flight: ensure SAM3D is alive
    _ensure_sam3d_healthy || { log "FATAL: Cannot start SAM3D. Aborting."; exit 1; }

    cd "${SAGE_DIR}/client"

    STAGE13_LOG="/tmp/sage_stage13_$(date +%s).log"
    log "Running robot task client... (log: ${STAGE13_LOG})"

    export PHYSICS_CRITIC_ENABLED
    SERVER_PATHS=(../server/layout.py)
    if [[ "${PHYSICS_CRITIC_ENABLED}" == "1" || "${PHYSICS_CRITIC_ENABLED}" == "true" ]]; then
        SERVER_PATHS+=(../server/physics/physics.py)
    fi

    # Background watchdog: restart SAM3D if it dies during generation
    (
        while true; do
            sleep 30
            wd_status=$(curl -sf -o /dev/null -w "%{http_code}" "${SAM3D_URL}/health" 2>/dev/null || echo "000")
            if [[ "${wd_status}" != "200" ]]; then
                echo "[pipeline $(date -u +%FT%TZ)] SAM3D watchdog: unhealthy (${wd_status}), restarting..." >&2
                pkill -f sam3d_server.py 2>/dev/null || true
                sleep 3
                wd_py="${PY_SYS}"
                [[ -x "/workspace/miniconda3/envs/sage/bin/python" ]] && wd_py="/workspace/miniconda3/envs/sage/bin/python"
                wd_args="--port ${SAM3D_PORT} --image-backend ${SAM3D_IMAGE_BACKEND:-gemini}"
                [[ -f /workspace/sam3d/checkpoints/hf/pipeline.yaml ]] && wd_args="${wd_args} --checkpoint-dir /workspace/sam3d/checkpoints/hf"
                nohup "${wd_py}" "${SAGE_SCRIPTS}/sam3d_server.py" ${wd_args} >> /tmp/sam3d_server.log 2>&1 &
                echo $! > /tmp/sam3d.pid
                for _w in $(seq 1 100); do
                    curl -sf "${SAM3D_URL}/health" >/dev/null 2>&1 && break
                    sleep 3
                done
                echo "[pipeline $(date -u +%FT%TZ)] SAM3D watchdog: restart complete" >&2
            fi
        done
    ) &
    SAM3D_WATCHDOG_PID=$!
    _stop_sam3d_watchdog() {
        if [[ -n "${SAM3D_WATCHDOG_PID:-}" ]]; then
            kill "${SAM3D_WATCHDOG_PID}" 2>/dev/null || true
            wait "${SAM3D_WATCHDOG_PID}" 2>/dev/null || true
            SAM3D_WATCHDOG_PID=""
        fi
    }
    trap _stop_sam3d_watchdog EXIT

    # Run scene generation with timeout
    set +e
    if command -v timeout >/dev/null 2>&1; then
        timeout "${SCENE_GEN_TIMEOUT}" \
            python client_generation_robot_task.py \
                --room_type "${ROOM_TYPE}" \
                --robot_type "${ROBOT_TYPE}" \
                --task_description "${TASK_DESC_CAPPED}" \
                --server_paths "${SERVER_PATHS[@]}" \
                2>&1 | tee "${STAGE13_LOG}"
        SCENE_GEN_EXIT=${PIPESTATUS[0]}
    else
        log "WARNING: 'timeout' not found; running scene generation without an outer timeout."
        python client_generation_robot_task.py \
            --room_type "${ROOM_TYPE}" \
            --robot_type "${ROBOT_TYPE}" \
            --task_description "${TASK_DESC_CAPPED}" \
            --server_paths "${SERVER_PATHS[@]}" \
            2>&1 | tee "${STAGE13_LOG}"
        SCENE_GEN_EXIT=${PIPESTATUS[0]}
    fi
    set -e

    # Stop watchdog (now that mesh generation is over)
    _stop_sam3d_watchdog

    if [[ "${SCENE_GEN_EXIT}" -eq 124 ]]; then
        log "ERROR: Scene generation timed out after ${SCENE_GEN_TIMEOUT}s"
        exit 1
    elif [[ "${SCENE_GEN_EXIT}" -ne 0 ]]; then
        if [[ "${STRICT_PIPELINE}" == "1" ]]; then
            log "ERROR: Scene generation exited with code ${SCENE_GEN_EXIT} (STRICT_PIPELINE=1)"
            exit 1
        fi
        log "WARNING: Scene generation exited with code ${SCENE_GEN_EXIT} (continuing; STRICT_PIPELINE=0)"
    fi

    # Find the latest layout directory
    LAYOUT_ID=$(ls -td "${SAGE_DIR}/server/results/layout_"* 2>/dev/null | head -1 | xargs basename)
    if [[ -z "${LAYOUT_ID}" ]]; then
        log "ERROR: No layout directory found after scene generation"
        exit 1
    fi
    LAYOUT_DIR="${SAGE_DIR}/server/results/${LAYOUT_ID}"
fi

log "Layout: ${LAYOUT_ID}"
log "Directory: ${LAYOUT_DIR}"

if [[ ! -d "${LAYOUT_DIR}" ]]; then
    log "ERROR: Layout directory does not exist: ${LAYOUT_DIR}"
    exit 1
fi

# Count generated objects
NUM_OBJECTS=$(ls "${LAYOUT_DIR}/generation/"*.obj 2>/dev/null | wc -l || echo "0")
log "Objects generated: ${NUM_OBJECTS}"

if [[ "${NUM_OBJECTS}" -eq 0 ]]; then
    if [[ "${STRICT_PIPELINE}" == "1" ]]; then
        log "ERROR: Zero objects generated in ${LAYOUT_DIR}/generation/. Aborting (STRICT_PIPELINE=1)."
        exit 1
    fi
    log "WARNING: Zero objects generated. Downstream stages may fail."
fi

STAGE13_END=$(date +%s)
log "Scene generation completed in $(( STAGE13_END - PIPELINE_START ))s"

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
	            # Stage 4a is best-effort: it improves scene richness but is not required
	            # for pose augmentation or downstream grasp/planning stages.
	            log "WARNING: MCP augmentation failed (continuing; Stage 4a is best-effort)."
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
	    POSE_AUG_SCRIPT="${SAGE_DIR}/server/augment/pose_aug_mm_from_layout_with_task.py"
	    if [[ -f "${POSE_AUG_SCRIPT}" ]]; then
	        if ! (cd "${SAGE_DIR}/server" && export PYTHONPATH="${SAGE_DIR}/server:${PYTHONPATH:-}" && python "${POSE_AUG_SCRIPT}" \
	            --layout_id "${LAYOUT_ID}" \
	            --save_dir_name "${POSE_AUG_NAME}" \
	            --num_samples "${NUM_POSE_SAMPLES}" \
	            2>&1 | tee "/tmp/sage_stage4b.log"); then
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

    STAGE4_END=$(date +%s)
    log "Stage 4 completed in $(( STAGE4_END - STAGE13_END ))s"
else
    log ""
    log "════ STAGE 4: SKIPPED (SKIP_AUGMENTATION=1) ════"
    STAGE4_END=$(date +%s)
fi

# ── Validate pose-aug meta (required for stages 5-7) ───────────────────────
if [[ "${SKIP_GRASPS}" != "1" && "${SKIP_DATA_GEN}" != "1" ]]; then
    POSE_META="${LAYOUT_DIR}/${POSE_AUG_NAME}/meta.json"
    if [[ ! -f "${POSE_META}" ]]; then
        log "ERROR: Missing pose augmentation meta: ${POSE_META}"
        exit 1
    fi
    NUM_LAYOUTS_IN_META="$("${PY_SYS}" - <<PY
import json
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
        log "ERROR: ${POSE_META} contains 0 layouts"
        exit 1
    fi
    log "Pose meta OK: ${NUM_LAYOUTS_IN_META} variants (${POSE_META})"
fi

# ── Quality gates + bounded repairs (SceneSmith-style) ─────────────────────
if [[ "${SKIP_GRASPS}" != "1" && "${SKIP_DATA_GEN}" != "1" ]]; then
    log ""
    log "════ QUALITY: Scene Repair + Gates ════"
    QUALITY_LOG="/tmp/sage_scene_quality_$(date +%s).log"
    QUALITY_SCRIPT="${SAGE_SCRIPTS}/sage_scene_quality.py"
    if [[ ! -f "${QUALITY_SCRIPT}" ]]; then
        log "ERROR: Missing quality script: ${QUALITY_SCRIPT}"
        exit 1
    fi

    if "${PY_SYS}" "${QUALITY_SCRIPT}" \
        --layout_dir "${LAYOUT_DIR}" \
        --pose_aug_name "${POSE_AUG_NAME}" \
        --profile "${SAGE_QUALITY_PROFILE}" \
        --max_iters "${SAGE_QUALITY_MAX_ITERS}" \
        2>&1 | tee "${QUALITY_LOG}"; then
        log "Quality gates: PASS"
    else
        log "Quality gates: FAIL (log: ${QUALITY_LOG})"

        if [[ "${SAGE_QUALITY_FALLBACK_ENABLED}" == "1" && "${SCENE_SOURCE}" == "sage" ]]; then
            log "Falling back to SceneSmith → SAGE for this scene..."
            FALLBACK_LOG="/tmp/scenesmith_fallback_$(date +%s).log"
            SCENESMITH_TO_SAGE="${SAGE_SCRIPTS}/scenesmith_to_sage_layout.py"
            LAYOUT_ID="$("${PY_SYS}" "${SCENESMITH_TO_SAGE}" \
                --results_dir "${SAGE_DIR}/server/results" \
                --room_type "${ROOM_TYPE}" \
                --task_desc "${TASK_DESC_CAPPED}" \
                --pose_aug_name "${POSE_AUG_NAME}" \
                2> >(tee "${FALLBACK_LOG}" >&2))"
            echo "${LAYOUT_ID}" >> "${FALLBACK_LOG}"
            if [[ -z "${LAYOUT_ID}" ]]; then
                log "ERROR: SceneSmith fallback did not return a layout_id"
                exit 1
            fi
            LAYOUT_DIR="${SAGE_DIR}/server/results/${LAYOUT_ID}"
            SCENE_SOURCE="scenesmith"

            NUM_OBJECTS=$(ls "${LAYOUT_DIR}/generation/"*.obj 2>/dev/null | wc -l || echo "0")
            log "Fallback layout: ${LAYOUT_ID} (objects=${NUM_OBJECTS})"

            # Validate meta again.
            POSE_META="${LAYOUT_DIR}/${POSE_AUG_NAME}/meta.json"
            if [[ ! -f "${POSE_META}" ]]; then
                log "ERROR: Fallback missing pose meta: ${POSE_META}"
                exit 1
            fi

            # Re-run quality on fallback.
            if ! "${PY_SYS}" "${QUALITY_SCRIPT}" \
                --layout_dir "${LAYOUT_DIR}" \
                --pose_aug_name "${POSE_AUG_NAME}" \
                --profile "${SAGE_QUALITY_PROFILE}" \
                --max_iters "${SAGE_QUALITY_MAX_ITERS}" \
                2>&1 | tee "/tmp/sage_scene_quality_fallback.log"; then
                log "ERROR: Quality gates still failing after SceneSmith fallback."
                exit 1
            fi
            log "Quality gates: PASS after fallback"
        else
            if [[ "${STRICT_PIPELINE}" == "1" ]]; then
                log "ERROR: Strict mode and quality gates failed (fallback disabled or unavailable)."
                exit 1
            fi
            log "WARNING: Quality gates failed (non-fatal; STRICT_PIPELINE=0)."
        fi
    fi
fi

# ── Unload SAM3D (free ~13GB VRAM for stages 5-7) ──────────────────────────
if [[ "${SKIP_GRASPS}" != "1" && "${SKIP_DATA_GEN}" != "1" ]]; then
    log ""
    log "Unloading SAM3D to free VRAM..."
    curl -sf -X POST http://localhost:8080/shutdown 2>/dev/null || \
        pkill -f sam3d_server.py 2>/dev/null || true
    sleep 3
    log "SAM3D unloaded"
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

    STAGE567_ARGS=(--layout_id "${LAYOUT_ID}" --results_dir "${SAGE_DIR}/server/results" --pose_aug_name "${POSE_AUG_NAME}" --num_demos "${NUM_DEMOS}")
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

# ── Interactive Backends (optional) ───────────────────────────────────────
if [[ "${ENABLE_INTERACTIVE}" == "1" ]]; then
    log ""
    log "════ INTERACTIVE BACKENDS ════"

    BACKENDS_SCRIPT="${SAGE_SCRIPTS}/start_interactive_backends.sh"
    if [[ ! -f "${BACKENDS_SCRIPT}" ]]; then
        log "ERROR: Missing ${BACKENDS_SCRIPT}"
        log "  Run install_interactive_backends.sh first to set up backends."
        exit 1
    fi

    bash "${BACKENDS_SCRIPT}" all 2>&1 | tee "/tmp/interactive_backends.log"

    # Wait for backends to become healthy (up to 60s)
    log "Waiting for backends to become healthy..."
    BACKEND_READY=0
    for _i in $(seq 1 12); do
        PHYSX_OK=0
        INFIN_OK=0
        if curl -sf http://localhost:8083/ >/dev/null 2>&1; then PHYSX_OK=1; fi
        if curl -sf http://localhost:8084/ >/dev/null 2>&1; then INFIN_OK=1; fi

        if [[ ${PHYSX_OK} -eq 1 || ${INFIN_OK} -eq 1 ]]; then
            BACKEND_READY=1
            break
        fi
        sleep 5
    done

    if [[ ${BACKEND_READY} -eq 1 ]]; then
        log "Interactive backends ready:"
        [[ ${PHYSX_OK:-0} -eq 1 ]] && log "  PhysX-Anything: http://localhost:8083"
        [[ ${INFIN_OK:-0} -eq 1 ]] && log "  Infinigen:      http://localhost:8084"

        # Export env vars for downstream tools
        export ARTICULATION_BACKEND=auto
        export PHYSX_ANYTHING_ENABLED=true
        export PHYSX_ANYTHING_ENDPOINT=http://localhost:8083
        export INFINIGEN_ENABLED=true
        export INFINIGEN_ENDPOINT=http://localhost:8084
    else
        log "WARNING: No interactive backends responded within 60s"
        log "  Check logs: /tmp/physx_anything_service.log, /tmp/infinigen_service.log"
    fi

    INTERACTIVE_END=$(date +%s)
    log "Interactive backends stage completed in $(( INTERACTIVE_END - BP_END ))s"
else
    INTERACTIVE_END=$(date +%s)
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
if [[ "${ENABLE_INTERACTIVE}" == "1" ]] && [[ ${BACKEND_READY:-0} -eq 1 ]]; then
    log "Interactive backends:"
    log "  ARTICULATION_BACKEND=auto"
    log "  PHYSX_ANYTHING_ENDPOINT=http://localhost:8083"
    log "  INFINIGEN_ENDPOINT=http://localhost:8084"
    log ""
fi
log "Logs:"
log "  Stages 1-3: ${STAGE13_LOG}"
log "  Stage 4:    /tmp/sage_stage4a.log, /tmp/sage_stage4b.log"
log "  Quality:    ${QUALITY_LOG:-}"
log "  Stages 5-7: /tmp/sage_stage567.log"
log "  BP:         /tmp/sage_bp_postprocess.log"
if [[ "${ENABLE_INTERACTIVE}" == "1" ]]; then
    log "  Backends:   /tmp/interactive_backends.log"
    log "  PhysX-Any:  /tmp/physx_anything_service.log"
    log "  Infinigen:  /tmp/infinigen_service.log"
fi
log "=========================================="
