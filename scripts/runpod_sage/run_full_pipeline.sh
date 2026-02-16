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
# Navigation robustness controls (Stage 4b pre-conditioning + retry).
NAV_CORRIDOR_REPAIR="${NAV_CORRIDOR_REPAIR:-1}"                 # 1|0
NAV_MIN_CORRIDOR_M="${NAV_MIN_CORRIDOR_M:-0.95}"                # meters
NAV_MIN_CORRIDOR_RETRY_M="${NAV_MIN_CORRIDOR_RETRY_M:-1.10}"    # meters
NAV_REPAIR_MAX_MOVES="${NAV_REPAIR_MAX_MOVES:-8}"               # objects
# Hard robot-nav gate (pre-acceptance).
NAV_GATE_ENABLED="${NAV_GATE_ENABLED:-1}"                       # 1|0
NAV_GATE_HARD_FAIL="${NAV_GATE_HARD_FAIL:-1}"                   # 1|0
NAV_GATE_GRID_RES_M="${NAV_GATE_GRID_RES_M:-0.05}"              # meters
NAV_GATE_PICK_RADIUS_MIN_M="${NAV_GATE_PICK_RADIUS_MIN_M:-0.55}"# meters
NAV_GATE_PICK_RADIUS_MAX_M="${NAV_GATE_PICK_RADIUS_MAX_M:-1.40}"# meters
# SceneSmith critic-loop acceptance gate (pre-mesh, bounded retries).
SCENESMITH_CRITIC_LOOP_ENABLED="${SCENESMITH_CRITIC_LOOP_ENABLED:-1}"             # 1|0
SCENESMITH_CRITIC_MAX_ATTEMPTS="${SCENESMITH_CRITIC_MAX_ATTEMPTS:-4}"              # attempts
SCENESMITH_CRITIC_REQUIRE_QUALITY_PASS="${SCENESMITH_CRITIC_REQUIRE_QUALITY_PASS:-1}"  # 1|0
SCENESMITH_CRITIC_REQUIRE_SCORE="${SCENESMITH_CRITIC_REQUIRE_SCORE:-1}"            # 1|0
SCENESMITH_CRITIC_REQUIRE_FAITHFULNESS="${SCENESMITH_CRITIC_REQUIRE_FAITHFULNESS:-1}"  # 1|0
SCENESMITH_CRITIC_MIN_SCORE_0_10="${SCENESMITH_CRITIC_MIN_SCORE_0_10:-8.0}"
SCENESMITH_CRITIC_MIN_SCORE_0_1="${SCENESMITH_CRITIC_MIN_SCORE_0_1:-0.80}"
SCENESMITH_CRITIC_MIN_FAITHFULNESS="${SCENESMITH_CRITIC_MIN_FAITHFULNESS:-0.80}"
SCENESMITH_CRITIC_SEED_STRIDE="${SCENESMITH_CRITIC_SEED_STRIDE:-7919}"
SCENESMITH_CRITIC_ALLOW_LAST_ATTEMPT_ON_FAIL="${SCENESMITH_CRITIC_ALLOW_LAST_ATTEMPT_ON_FAIL:-0}"  # 1|0
# SAGE scene acceptance retry loop (pre-Stage4 quality+nav gate on Stage 1-3 output).
SAGE_STAGE13_CRITIC_LOOP_ENABLED="${SAGE_STAGE13_CRITIC_LOOP_ENABLED:-1}"  # 1|0
SAGE_STAGE13_MAX_ATTEMPTS="${SAGE_STAGE13_MAX_ATTEMPTS:-3}"                # attempts
SAGE_STAGE13_CRITIC_MIN_TOTAL_0_10="${SAGE_STAGE13_CRITIC_MIN_TOTAL_0_10:-8.0}"
SAGE_STAGE13_CRITIC_MIN_FAITHFULNESS="${SAGE_STAGE13_CRITIC_MIN_FAITHFULNESS:-0.80}"
SAGE_STAGE13_CRITIC_MAX_COLLISION_RATE="${SAGE_STAGE13_CRITIC_MAX_COLLISION_RATE:-0.08}"
# NOTE: SAGE's physics MCP server starts Isaac Sim (noisy stdout). If enabled, ensure your MCP stdio stack
# is patched to ignore non-JSON stdout and Isaac Sim can run headless. Default off for robustness.
PHYSICS_CRITIC_ENABLED="${PHYSICS_CRITIC_ENABLED:-false}"

# Resume from an existing results directory (skips stages 1-3 scene generation).
# Set to an explicit layout id (e.g. layout_12345) or "latest".
RESUME_LAYOUT_ID="${RESUME_LAYOUT_ID:-}"

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
log "SceneSmith critic loop: enabled=${SCENESMITH_CRITIC_LOOP_ENABLED} attempts=${SCENESMITH_CRITIC_MAX_ATTEMPTS} qg=${SCENESMITH_CRITIC_REQUIRE_QUALITY_PASS} score=${SCENESMITH_CRITIC_REQUIRE_SCORE} faith=${SCENESMITH_CRITIC_REQUIRE_FAITHFULNESS}"
log "SAGE Stage1-3 accept loop: enabled=${SAGE_STAGE13_CRITIC_LOOP_ENABLED} attempts=${SAGE_STAGE13_MAX_ATTEMPTS}"
log "SAGE pre-Stage4 critic thresholds: total>=${SAGE_STAGE13_CRITIC_MIN_TOTAL_0_10}, faith>=${SAGE_STAGE13_CRITIC_MIN_FAITHFULNESS}, collision<=${SAGE_STAGE13_CRITIC_MAX_COLLISION_RATE}"
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

_run_sage_stage13_once() {
    local _attempt_idx="${1:-1}"
    local _stage13_log="${2:-/tmp/sage_stage13.log}"

    _ensure_sam3d_healthy || {
        log "ERROR: Cannot start SAM3D before Stage 1-3."
        return 1
    }

    cd "${SAGE_DIR}/client"

    export PHYSICS_CRITIC_ENABLED
    local _task_desc="${TASK_DESC_CAPPED}"
    if [[ "${_attempt_idx}" -gt 1 ]]; then
        _task_desc="${TASK_DESC_CAPPED} Generate a different valid scene arrangement than previous attempts while keeping task-critical objects."
    fi

    local -a _server_paths=(../server/layout.py)
    if [[ "${PHYSICS_CRITIC_ENABLED}" == "1" || "${PHYSICS_CRITIC_ENABLED}" == "true" ]]; then
        _server_paths+=(../server/physics/physics.py)
    fi

    # Background watchdog: restart SAM3D if it dies during generation.
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
    local _watchdog_pid=$!

    set +e
    local _scene_gen_exit=0
    if command -v timeout >/dev/null 2>&1; then
        timeout "${SCENE_GEN_TIMEOUT}" \
            python client_generation_robot_task.py \
                --room_type "${ROOM_TYPE}" \
                --robot_type "${ROBOT_TYPE}" \
                --task_description "${_task_desc}" \
                --server_paths "${_server_paths[@]}" \
                2>&1 | tee "${_stage13_log}"
        _scene_gen_exit=${PIPESTATUS[0]}
    else
        log "WARNING: 'timeout' not found; running scene generation without an outer timeout."
        python client_generation_robot_task.py \
            --room_type "${ROOM_TYPE}" \
            --robot_type "${ROBOT_TYPE}" \
            --task_description "${_task_desc}" \
            --server_paths "${_server_paths[@]}" \
            2>&1 | tee "${_stage13_log}"
        _scene_gen_exit=${PIPESTATUS[0]}
    fi
    set -e

    kill "${_watchdog_pid}" 2>/dev/null || true
    wait "${_watchdog_pid}" 2>/dev/null || true

    if [[ "${_scene_gen_exit}" -eq 124 ]]; then
        log "ERROR: Scene generation timed out after ${SCENE_GEN_TIMEOUT}s"
        return 124
    fi
    if [[ "${_scene_gen_exit}" -ne 0 ]]; then
        if [[ "${STRICT_PIPELINE}" == "1" ]]; then
            log "ERROR: Scene generation exited with code ${_scene_gen_exit} (STRICT_PIPELINE=1)"
            return "${_scene_gen_exit}"
        fi
        log "WARNING: Scene generation exited with code ${_scene_gen_exit} (continuing; STRICT_PIPELINE=0)"
    fi

    # Find the latest layout directory.
    LAYOUT_ID=$(ls -td "${SAGE_DIR}/server/results/layout_"* 2>/dev/null | head -1 | xargs basename)
    if [[ -z "${LAYOUT_ID}" ]]; then
        log "ERROR: No layout directory found after scene generation"
        return 1
    fi
    LAYOUT_DIR="${SAGE_DIR}/server/results/${LAYOUT_ID}"
    return 0
}

_prepare_layout_for_stage4() {
    log "Layout: ${LAYOUT_ID}"
    log "Directory: ${LAYOUT_DIR}"

    if [[ ! -d "${LAYOUT_DIR}" ]]; then
        log "ERROR: Layout directory does not exist: ${LAYOUT_DIR}"
        return 1
    fi

    # Count generated objects.
    NUM_OBJECTS=$(ls "${LAYOUT_DIR}/generation/"*.obj 2>/dev/null | wc -l || echo "0")
    log "Objects generated: ${NUM_OBJECTS}"

    # Repair layout JSON for downstream stage compatibility (4a/4b).
    LAYOUT_JSON="${LAYOUT_DIR}/${LAYOUT_ID}.json"
    if [[ ! -f "${LAYOUT_JSON}" ]]; then
        log "ERROR: Missing layout JSON: ${LAYOUT_JSON}"
        return 1
    fi
    FIX_LAYOUT_JSON_SCRIPT="${SAGE_SCRIPTS}/fix_layout_json.py"
    if [[ -f "${FIX_LAYOUT_JSON_SCRIPT}" ]]; then
        log "Repairing layout JSON for Stage 4 (policy_analysis matches + updated_task_decomposition)..."
        if ! "${PY_SYS}" "${FIX_LAYOUT_JSON_SCRIPT}" "${LAYOUT_JSON}" 2>&1 | tee "/tmp/sage_fix_layout_json.log"; then
            log "ERROR: Layout JSON repair failed (see /tmp/sage_fix_layout_json.log)"
            return 1
        fi
    else
        log "ERROR: Missing layout JSON repair script: ${FIX_LAYOUT_JSON_SCRIPT}"
        return 1
    fi

    NAV_REPAIR_SCRIPT="${SAGE_SCRIPTS}/repair_navigation_corridor.py"
    if [[ "${NAV_CORRIDOR_REPAIR}" == "1" ]]; then
        if [[ -f "${NAV_REPAIR_SCRIPT}" ]]; then
            log "Pre-conditioning layout for mobile-base corridor clearance..."
            if ! "${PY_SYS}" "${NAV_REPAIR_SCRIPT}" \
                --layout_json "${LAYOUT_JSON}" \
                --min_corridor_m "${NAV_MIN_CORRIDOR_M}" \
                --max_moves "${NAV_REPAIR_MAX_MOVES}" \
                2>&1 | tee "/tmp/sage_nav_repair_pre4.log"; then
                log "WARNING: navigation corridor pre-repair failed (continuing)."
            fi
        else
            log "WARNING: navigation repair script not found: ${NAV_REPAIR_SCRIPT}"
        fi
    fi

    NAV_GATE_SCRIPT="${SAGE_SCRIPTS}/robot_nav_gate.py"
    if [[ "${NAV_GATE_ENABLED}" == "1" ]]; then
        if [[ -f "${NAV_GATE_SCRIPT}" ]]; then
            NAV_GATE_REPORT="${LAYOUT_DIR}/quality/nav_gate_report.json"
            log "Running hard robot-nav gate before scene acceptance..."
            NAV_GATE_OK=0
            if ! "${PY_SYS}" "${NAV_GATE_SCRIPT}" \
                --layout_json "${LAYOUT_JSON}" \
                --report_path "${NAV_GATE_REPORT}" \
                --grid_res_m "${NAV_GATE_GRID_RES_M}" \
                --pick_radius_min_m "${NAV_GATE_PICK_RADIUS_MIN_M}" \
                --pick_radius_max_m "${NAV_GATE_PICK_RADIUS_MAX_M}" \
                2>&1 | tee "/tmp/sage_nav_gate_pre4.log"; then
                NAV_GATE_OK=1
            fi

            if [[ "${NAV_GATE_OK}" != "0" && "${NAV_CORRIDOR_REPAIR}" == "1" && -f "${NAV_REPAIR_SCRIPT}" ]]; then
                log "Robot-nav gate failed. Applying aggressive corridor repair and retrying nav gate once..."
                if "${PY_SYS}" "${NAV_REPAIR_SCRIPT}" \
                    --layout_json "${LAYOUT_JSON}" \
                    --min_corridor_m "${NAV_MIN_CORRIDOR_RETRY_M}" \
                    --max_moves "$(( NAV_REPAIR_MAX_MOVES * 2 ))" \
                    --aggressive \
                    2>&1 | tee "/tmp/sage_nav_repair_gate_retry.log"; then
                    NAV_GATE_OK=0
                    if ! "${PY_SYS}" "${NAV_GATE_SCRIPT}" \
                        --layout_json "${LAYOUT_JSON}" \
                        --report_path "${NAV_GATE_REPORT}" \
                        --grid_res_m "${NAV_GATE_GRID_RES_M}" \
                        --pick_radius_min_m "${NAV_GATE_PICK_RADIUS_MIN_M}" \
                        --pick_radius_max_m "${NAV_GATE_PICK_RADIUS_MAX_M}" \
                        2>&1 | tee "/tmp/sage_nav_gate_retry.log"; then
                        NAV_GATE_OK=1
                    fi
                fi
            fi

            if [[ "${NAV_GATE_OK}" != "0" ]]; then
                if [[ "${NAV_GATE_HARD_FAIL}" == "1" || "${STRICT_PIPELINE}" == "1" ]]; then
                    log "ERROR: Robot-nav gate failed. Rejecting scene before Stage 4/5."
                    log "  See report: ${NAV_GATE_REPORT}"
                    return 1
                fi
                log "WARNING: Robot-nav gate failed but continuing (NAV_GATE_HARD_FAIL=0, STRICT_PIPELINE=0)."
            fi
        else
            if [[ "${NAV_GATE_HARD_FAIL}" == "1" || "${STRICT_PIPELINE}" == "1" ]]; then
                log "ERROR: Robot-nav gate script missing: ${NAV_GATE_SCRIPT}"
                return 1
            fi
            log "WARNING: Robot-nav gate script missing (continuing)."
        fi
    fi

    if [[ "${NUM_OBJECTS}" -eq 0 ]]; then
        if [[ "${STRICT_PIPELINE}" == "1" ]]; then
            log "ERROR: Zero objects generated in ${LAYOUT_DIR}/generation/. Aborting (STRICT_PIPELINE=1)."
            return 1
        fi
        log "WARNING: Zero objects generated. Downstream stages may fail."
    fi
    return 0
}

_run_sage_prestage4_quality_gate() {
    local _critic_script="${SAGE_SCRIPTS}/sage_scenesmith_critic_gate.py"
    local _quality_script="${SAGE_SCRIPTS}/sage_scene_quality.py"
    if [[ ! -f "${_critic_script}" ]]; then
        log "ERROR: Missing SceneSmith critic gate script for SAGE acceptance loop: ${_critic_script}"
        return 1
    fi
    if [[ ! -f "${_quality_script}" ]]; then
        log "ERROR: Missing quality script for SAGE acceptance loop: ${_quality_script}"
        return 1
    fi

    local _critic_report="${LAYOUT_DIR}/quality/sage_scenesmith_critic_report.json"
    local _critic_log="/tmp/sage_scenesmith_critic_pre4_${LAYOUT_ID}_$(date +%s).log"
    if ! "${PY_SYS}" "${_critic_script}" \
        --layout_json "${LAYOUT_JSON}" \
        --task_desc "${TASK_DESC}" \
        --report_path "${_critic_report}" \
        --min_total_0_10 "${SAGE_STAGE13_CRITIC_MIN_TOTAL_0_10}" \
        --min_faithfulness "${SAGE_STAGE13_CRITIC_MIN_FAITHFULNESS}" \
        --max_collision_rate "${SAGE_STAGE13_CRITIC_MAX_COLLISION_RATE}" \
        2>&1 | tee "${_critic_log}"; then
        log "Pre-Stage4 SceneSmith critic gate: FAIL (log: ${_critic_log})"
        return 1
    fi

    local _quality_log="/tmp/sage_scene_quality_pre4_${LAYOUT_ID}_$(date +%s).log"
    if "${PY_SYS}" "${_quality_script}" \
        --layout_dir "${LAYOUT_DIR}" \
        --pose_aug_name "${POSE_AUG_NAME}" \
        --profile "${SAGE_QUALITY_PROFILE}" \
        --max_iters "${SAGE_QUALITY_MAX_ITERS}" \
        2>&1 | tee "${_quality_log}"; then
        log "Pre-Stage4 quality gates: PASS (SceneSmith critic + scene quality)"
        return 0
    fi
    log "Pre-Stage4 quality gate: FAIL (log: ${_quality_log})"
    return 1
}

# ── SCENE GENERATION ───────────────────────────────────────────────────────
LAYOUT_PREPARED=0
if [[ -n "${RESUME_LAYOUT_ID}" ]]; then
    log "════ RESUME: Using Existing Layout Results ════"
    if [[ "${RESUME_LAYOUT_ID}" == "latest" ]]; then
        LAYOUT_ID=$(ls -td "${SAGE_DIR}/server/results/layout_"* 2>/dev/null | head -1 | xargs basename)
        if [[ -z "${LAYOUT_ID}" ]]; then
            log "ERROR: RESUME_LAYOUT_ID=latest but no layout directory found in ${SAGE_DIR}/server/results/"
            exit 1
        fi
    else
        LAYOUT_ID="${RESUME_LAYOUT_ID}"
    fi
    LAYOUT_DIR="${SAGE_DIR}/server/results/${LAYOUT_ID}"
elif [[ "${SCENE_SOURCE}" == "scenesmith" ]]; then
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
        --critic_loop_enabled "${SCENESMITH_CRITIC_LOOP_ENABLED}" \
        --critic_max_attempts "${SCENESMITH_CRITIC_MAX_ATTEMPTS}" \
        --critic_require_quality_pass "${SCENESMITH_CRITIC_REQUIRE_QUALITY_PASS}" \
        --critic_require_score "${SCENESMITH_CRITIC_REQUIRE_SCORE}" \
        --critic_require_faithfulness "${SCENESMITH_CRITIC_REQUIRE_FAITHFULNESS}" \
        --critic_min_score_0_10 "${SCENESMITH_CRITIC_MIN_SCORE_0_10}" \
        --critic_min_score_0_1 "${SCENESMITH_CRITIC_MIN_SCORE_0_1}" \
        --critic_min_faithfulness "${SCENESMITH_CRITIC_MIN_FAITHFULNESS}" \
        --critic_seed_stride "${SCENESMITH_CRITIC_SEED_STRIDE}" \
        --critic_allow_last_attempt_on_fail "${SCENESMITH_CRITIC_ALLOW_LAST_ATTEMPT_ON_FAIL}" \
        2> >(tee "${STAGE13_LOG}" >&2))"
    echo "${LAYOUT_ID}" >> "${STAGE13_LOG}"
    if [[ -z "${LAYOUT_ID}" ]]; then
        log "ERROR: SceneSmith->SAGE bridge did not return a layout_id"
        exit 1
    fi
    LAYOUT_DIR="${SAGE_DIR}/server/results/${LAYOUT_ID}"
else
    log "════ STAGES 1-3: Scene Generation (SAGE) ════"
    SAGE_STAGE13_ATTEMPTS=1
    if [[ "${SAGE_STAGE13_CRITIC_LOOP_ENABLED}" == "1" ]]; then
        SAGE_STAGE13_ATTEMPTS="${SAGE_STAGE13_MAX_ATTEMPTS}"
    fi
    if [[ ! "${SAGE_STAGE13_ATTEMPTS}" =~ ^[0-9]+$ || "${SAGE_STAGE13_ATTEMPTS}" -lt 1 ]]; then
        log "WARNING: Invalid SAGE_STAGE13_MAX_ATTEMPTS='${SAGE_STAGE13_ATTEMPTS}'; defaulting to 1."
        SAGE_STAGE13_ATTEMPTS=1
    fi

    SAGE_STAGE13_ACCEPTED=0
    for _attempt in $(seq 1 "${SAGE_STAGE13_ATTEMPTS}"); do
        STAGE13_LOG="/tmp/sage_stage13_attempt${_attempt}_$(date +%s).log"
        log "Running robot task client... (attempt ${_attempt}/${SAGE_STAGE13_ATTEMPTS}, log: ${STAGE13_LOG})"

        if ! _run_sage_stage13_once "${_attempt}" "${STAGE13_LOG}"; then
            if [[ "${_attempt}" -lt "${SAGE_STAGE13_ATTEMPTS}" ]]; then
                log "SAGE Stage 1-3 attempt ${_attempt} failed. Regenerating..."
                continue
            fi
            log "ERROR: SAGE Stage 1-3 failed after ${SAGE_STAGE13_ATTEMPTS} attempts."
            exit 1
        fi

        if ! _prepare_layout_for_stage4; then
            if [[ "${_attempt}" -lt "${SAGE_STAGE13_ATTEMPTS}" ]]; then
                log "SAGE Stage 1-3 attempt ${_attempt} rejected by layout/nav gate. Regenerating..."
                continue
            fi
            log "ERROR: SAGE Stage 1-3 failed acceptance gates after ${SAGE_STAGE13_ATTEMPTS} attempts."
            exit 1
        fi

        if [[ "${SAGE_STAGE13_CRITIC_LOOP_ENABLED}" == "1" ]]; then
            if ! _run_sage_prestage4_quality_gate; then
                if [[ "${_attempt}" -lt "${SAGE_STAGE13_ATTEMPTS}" ]]; then
                    log "SAGE Stage 1-3 attempt ${_attempt} rejected by pre-Stage4 quality gate. Regenerating..."
                    continue
                fi
                log "ERROR: SAGE Stage 1-3 did not pass pre-Stage4 quality gate after ${SAGE_STAGE13_ATTEMPTS} attempts."
                exit 1
            fi
        fi

        SAGE_STAGE13_ACCEPTED=1
        LAYOUT_PREPARED=1
        break
    done

    if [[ "${SAGE_STAGE13_ACCEPTED}" != "1" ]]; then
        log "ERROR: SAGE Stage 1-3 acceptance loop exhausted without an accepted layout."
        exit 1
    fi
fi

if [[ "${LAYOUT_PREPARED}" != "1" ]]; then
    if ! _prepare_layout_for_stage4; then
        log "ERROR: Failed to prepare accepted layout for Stage 4."
        exit 1
    fi
    LAYOUT_PREPARED=1
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

		    # 4a sometimes rewrites the layout JSON; re-apply the compatibility patch so
		    # Stage 4b never fails on missing keys.
		    log "Re-checking layout JSON for Stage 4b..."
		    if ! "${PY_SYS}" "${FIX_LAYOUT_JSON_SCRIPT}" "${LAYOUT_JSON}" 2>&1 | tee "/tmp/sage_fix_layout_json_post4a.log"; then
		        log "ERROR: Layout JSON repair failed after Stage 4a (see /tmp/sage_fix_layout_json_post4a.log)"
		        exit 1
		    fi

		    # 4b: Pose augmentation
		    log "4b: Pose augmentation (${NUM_POSE_SAMPLES} samples)..."
		    POSE_AUG_SCRIPT="${SAGE_DIR}/server/augment/pose_aug_mm_from_layout_with_task.py"
		    if [[ -f "${POSE_AUG_SCRIPT}" ]]; then
		        _count_pose_meta_layouts() {
		            local _meta_path="$1"
		            if [[ ! -f "${_meta_path}" ]]; then
		                echo "0"
		                return 0
		            fi
		            "${PY_SYS}" - <<PY
import json
path = "${_meta_path}"
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
		        }

		        _run_pose_aug_for_dir() {
		            local _save_dir_name="$1"
		            local _log_file="$2"
		            (cd "${SAGE_DIR}/server" && export PYTHONPATH="${SAGE_DIR}/server:${PYTHONPATH:-}" && python "${POSE_AUG_SCRIPT}" \
		                --layout_id "${LAYOUT_ID}" \
		                --save_dir_name "${_save_dir_name}" \
		                --num_samples "${NUM_POSE_SAMPLES}" \
		                2>&1 | tee "${_log_file}")
		        }

		        if ! _run_pose_aug_for_dir "${POSE_AUG_NAME}" "/tmp/sage_stage4b.log"; then
		            if [[ "${STRICT_PIPELINE}" == "1" ]]; then
		                log "ERROR: Pose augmentation failed in strict mode."
		                exit 1
		            fi
		            log "WARNING: Pose augmentation failed (non-fatal)"
		        fi

		        # If planner filtering yields 0 feasible layouts, attempt one aggressive
		        # corridor repair + Stage 4b retry before synthetic fallback meta.
		        POSE_META_FALLBACK="${LAYOUT_DIR}/${POSE_AUG_NAME}/meta.json"
		        POSE_META_COUNT="$(_count_pose_meta_layouts "${POSE_META_FALLBACK}")"
		        if [[ "${POSE_META_COUNT}" -lt 1 && "${NAV_CORRIDOR_REPAIR}" == "1" && -f "${NAV_REPAIR_SCRIPT}" ]]; then
		            log "WARNING: Stage 4b produced 0 feasible layouts. Applying aggressive corridor repair and retrying once..."
		            if "${PY_SYS}" "${NAV_REPAIR_SCRIPT}" \
		                --layout_json "${LAYOUT_JSON}" \
		                --min_corridor_m "${NAV_MIN_CORRIDOR_RETRY_M}" \
		                --max_moves "$(( NAV_REPAIR_MAX_MOVES * 2 ))" \
		                --aggressive \
		                2>&1 | tee "/tmp/sage_nav_repair_retry4b.log"; then
		                POSE_AUG_RETRY_NAME="${POSE_AUG_NAME}_retry1"
		                if ! _run_pose_aug_for_dir "${POSE_AUG_RETRY_NAME}" "/tmp/sage_stage4b_retry.log"; then
		                    if [[ "${STRICT_PIPELINE}" == "1" ]]; then
		                        log "ERROR: Pose augmentation retry failed in strict mode."
		                        exit 1
		                    fi
		                    log "WARNING: Pose augmentation retry failed (non-fatal)"
		                fi
		                POSE_AUG_NAME="${POSE_AUG_RETRY_NAME}"
		                POSE_META_FALLBACK="${LAYOUT_DIR}/${POSE_AUG_NAME}/meta.json"
		                POSE_META_COUNT="$(_count_pose_meta_layouts "${POSE_META_FALLBACK}")"
		                log "Stage 4b retry complete (pose_aug_name=${POSE_AUG_NAME}, feasible_layouts=${POSE_META_COUNT})"
		            else
		                log "WARNING: aggressive corridor repair failed; continuing with fallback meta logic."
		            fi
		        fi

		        # Final fallback: if meta is missing or empty, synthesize from generated variants.
		        if [[ "${POSE_META_COUNT}" -lt 1 ]]; then
		            if ls "${LAYOUT_DIR}/${POSE_AUG_NAME}"/variant_*.json >/dev/null 2>&1; then
		                if [[ -f "${POSE_META_FALLBACK}" ]]; then
		                    log "WARNING: Pose augmentation meta has 0 layouts; rebuilding fallback meta from generated variants."
		                else
		                    log "WARNING: Pose augmentation did not write meta.json; creating fallback meta from generated variants."
		                fi
		                "${PY_SYS}" - <<PY
import glob
import json
import os

pose_dir = "${LAYOUT_DIR}/${POSE_AUG_NAME}"
variant_paths = sorted(glob.glob(os.path.join(pose_dir, "variant_*.json")))
variant_names = [os.path.basename(p) for p in variant_paths]
# Use one canonical variant to keep strict Stage 5-7 deterministic/retryable.
pick = "variant_000.json" if "variant_000.json" in variant_names else (variant_names[0] if variant_names else None)
variants = [pick] if pick else []
out_path = os.path.join(pose_dir, "meta.json")
with open(out_path, "w") as f:
    json.dump(variants, f, indent=2)
print(f"[bp] wrote {out_path} ({len(variants)} variant, selected={pick})")
PY
		            fi
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
                --critic_loop_enabled "${SCENESMITH_CRITIC_LOOP_ENABLED}" \
                --critic_max_attempts "${SCENESMITH_CRITIC_MAX_ATTEMPTS}" \
                --critic_require_quality_pass "${SCENESMITH_CRITIC_REQUIRE_QUALITY_PASS}" \
                --critic_require_score "${SCENESMITH_CRITIC_REQUIRE_SCORE}" \
                --critic_require_faithfulness "${SCENESMITH_CRITIC_REQUIRE_FAITHFULNESS}" \
                --critic_min_score_0_10 "${SCENESMITH_CRITIC_MIN_SCORE_0_10}" \
                --critic_min_score_0_1 "${SCENESMITH_CRITIC_MIN_SCORE_0_1}" \
                --critic_min_faithfulness "${SCENESMITH_CRITIC_MIN_FAITHFULNESS}" \
                --critic_seed_stride "${SCENESMITH_CRITIC_SEED_STRIDE}" \
                --critic_allow_last_attempt_on_fail "${SCENESMITH_CRITIC_ALLOW_LAST_ATTEMPT_ON_FAIL}" \
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

            # Re-run hard robot-nav gate on the fallback scene before accepting it.
            LAYOUT_JSON="${LAYOUT_DIR}/${LAYOUT_ID}.json"
            if [[ "${NAV_GATE_ENABLED}" == "1" && -f "${NAV_GATE_SCRIPT}" ]]; then
                NAV_GATE_REPORT="${LAYOUT_DIR}/quality/nav_gate_report.json"
                NAV_GATE_OK=0
                log "Running hard robot-nav gate on fallback scene..."
                if ! "${PY_SYS}" "${NAV_GATE_SCRIPT}" \
                    --layout_json "${LAYOUT_JSON}" \
                    --report_path "${NAV_GATE_REPORT}" \
                    --grid_res_m "${NAV_GATE_GRID_RES_M}" \
                    --pick_radius_min_m "${NAV_GATE_PICK_RADIUS_MIN_M}" \
                    --pick_radius_max_m "${NAV_GATE_PICK_RADIUS_MAX_M}" \
                    2>&1 | tee "/tmp/sage_nav_gate_fallback.log"; then
                    NAV_GATE_OK=1
                fi
                if [[ "${NAV_GATE_OK}" != "0" && "${NAV_CORRIDOR_REPAIR}" == "1" && -f "${NAV_REPAIR_SCRIPT}" ]]; then
                    log "Fallback robot-nav gate failed. Applying aggressive corridor repair and retrying once..."
                    if "${PY_SYS}" "${NAV_REPAIR_SCRIPT}" \
                        --layout_json "${LAYOUT_JSON}" \
                        --min_corridor_m "${NAV_MIN_CORRIDOR_RETRY_M}" \
                        --max_moves "$(( NAV_REPAIR_MAX_MOVES * 2 ))" \
                        --aggressive \
                        2>&1 | tee "/tmp/sage_nav_repair_fallback_retry.log"; then
                        NAV_GATE_OK=0
                        if ! "${PY_SYS}" "${NAV_GATE_SCRIPT}" \
                            --layout_json "${LAYOUT_JSON}" \
                            --report_path "${NAV_GATE_REPORT}" \
                            --grid_res_m "${NAV_GATE_GRID_RES_M}" \
                            --pick_radius_min_m "${NAV_GATE_PICK_RADIUS_MIN_M}" \
                            --pick_radius_max_m "${NAV_GATE_PICK_RADIUS_MAX_M}" \
                            2>&1 | tee "/tmp/sage_nav_gate_fallback_retry.log"; then
                            NAV_GATE_OK=1
                        fi
                    fi
                fi
                if [[ "${NAV_GATE_OK}" != "0" ]]; then
                    log "ERROR: Fallback scene failed hard robot-nav gate."
                    exit 1
                fi
            fi

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
[[ -n "${NAV_GATE_REPORT:-}" ]] && log "  Nav gate: ${NAV_GATE_REPORT}"
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
log "  Nav gate:   /tmp/sage_nav_gate_pre4.log, /tmp/sage_nav_gate_retry.log"
log "  Nav fallback:/tmp/sage_nav_gate_fallback.log, /tmp/sage_nav_gate_fallback_retry.log"
log "  Quality:    ${QUALITY_LOG:-}"
log "  Stages 5-7: /tmp/sage_stage567.log"
log "  BP:         /tmp/sage_bp_postprocess.log"
if [[ "${ENABLE_INTERACTIVE}" == "1" ]]; then
    log "  Backends:   /tmp/interactive_backends.log"
    log "  PhysX-Any:  /tmp/physx_anything_service.log"
    log "  Infinigen:  /tmp/infinigen_service.log"
fi
log "=========================================="
