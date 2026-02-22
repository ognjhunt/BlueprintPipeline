#!/usr/bin/env bash
# =============================================================================
# GCS-Synced Pipeline Runner (latest-wins by object generation)
# =============================================================================
#
# Usage:
#   bash run_pipeline_gcs.sh <scene_id> <bucket> <object_name> <generation> [steps]
#
# Example:
#   bash run_pipeline_gcs.sh \
#     ChIJIfs1HWblrIkRV4sxRLX4sYM \
#     blueprint-8c1ca.appspot.com \
#     scenes/ChIJIfs1HWblrIkRV4sxRLX4sYM/images/living_room.jpeg \
#     1739220750732768
#
# Default steps:
#   text-scene-gen,text-scene-adapter
# =============================================================================

set -euo pipefail

SCENE_ID="${1:?Usage: $0 <scene_id> <bucket> <object_name> <generation> [steps]}"
BUCKET="${2:?Usage: $0 <scene_id> <bucket> <object_name> <generation> [steps]}"
OBJECT_NAME="${3:?Usage: $0 <scene_id> <bucket> <object_name> <generation> [steps]}"
OBJECT_GENERATION="${4:?Usage: $0 <scene_id> <bucket> <object_name> <generation> [steps]}"
STEPS="${5:-text-scene-gen,text-scene-adapter}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENE_DIR="/tmp/blueprint_scenes/${SCENE_ID}"
LOG_FILE="/tmp/blueprint_pipeline_${SCENE_ID}.log"

STATE_ROOT="/tmp/blueprint_pipeline_state"
SCENE_HASH="$(printf '%s' "${SCENE_ID}" | shasum -a 256 | awk '{print $1}')"
STATE_FILE="${STATE_ROOT}/${SCENE_HASH}.state"
LOCK_DIR="${STATE_ROOT}/${SCENE_HASH}.lock"

mkdir -p "${STATE_ROOT}"

timestamp_utc() {
    date -u +%Y-%m-%dT%H:%M:%SZ
}

preflight_runner_dependencies() {
    if ! command -v python3 >/dev/null 2>&1; then
        echo "[GCS-Runner] ERROR: python3 is not installed or not on PATH."
        echo "[GCS-Runner] Install Python 3 and retry."
        return 1
    fi

    local missing_modules
    set +e
    missing_modules="$(
        python3 - <<'PY'
import importlib.util

required = ("numpy", "yaml")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print(",".join(missing))
    raise SystemExit(1)
PY
    )"
    local rc=$?
    set -e
    if [ "${rc}" -ne 0 ]; then
        echo "[GCS-Runner] ERROR: Local runner dependency preflight failed."
        if [ -n "${missing_modules}" ]; then
            echo "[GCS-Runner] Missing Python modules: ${missing_modules}"
        fi
        echo "[GCS-Runner] Install required modules on this host and retry:"
        echo "  python3 -m pip install numpy PyYAML"
        return "${rc}"
    fi

    return 0
}

lock_state() {
    local attempts=0
    while ! mkdir "${LOCK_DIR}" 2>/dev/null; do
        attempts=$((attempts + 1))
        if [ "${attempts}" -ge 180 ]; then
            echo "[GCS-Runner] ERROR: Timed out waiting for scene lock ${LOCK_DIR}" >&2
            exit 1
        fi
        sleep 1
    done
}

unlock_state() {
    rmdir "${LOCK_DIR}" 2>/dev/null || true
}

compare_generation() {
    python3 - "$1" "$2" <<'PY'
import sys
a = sys.argv[1]
b = sys.argv[2]
try:
    ai = int(a)
    bi = int(b)
except ValueError:
    if a == b:
        print(0)
    elif a > b:
        print(1)
    else:
        print(-1)
else:
    print((ai > bi) - (ai < bi))
PY
}

read_state() {
    CURRENT_SCENE_ID=""
    CURRENT_GENERATION=""
    CURRENT_PID=""
    CURRENT_STATUS=""
    CURRENT_UPDATED_AT=""
    CURRENT_OBJECT_NAME=""

    if [ ! -f "${STATE_FILE}" ]; then
        return 0
    fi

    while IFS='=' read -r key value; do
        case "${key}" in
            scene_id) CURRENT_SCENE_ID="${value}" ;;
            generation) CURRENT_GENERATION="${value}" ;;
            pid) CURRENT_PID="${value}" ;;
            status) CURRENT_STATUS="${value}" ;;
            updated_at) CURRENT_UPDATED_AT="${value}" ;;
            object_name) CURRENT_OBJECT_NAME="${value}" ;;
        esac
    done < "${STATE_FILE}"
}

write_state() {
    local generation="${1}"
    local pid="${2}"
    local status="${3}"
    cat > "${STATE_FILE}" <<EOF
scene_id=${SCENE_ID}
generation=${generation}
pid=${pid}
status=${status}
updated_at=$(timestamp_utc)
object_name=${OBJECT_NAME}
EOF
}

process_running() {
    local pid="${1}"
    [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null
}

stop_existing_pid() {
    local pid="${1}"
    if [ -z "${pid}" ]; then
        return 0
    fi
    if ! process_running "${pid}"; then
        return 0
    fi

    echo "[GCS-Runner] Stopping stale pipeline PID ${pid} (superseded by newer generation)..."
    kill "${pid}" 2>/dev/null || true
    sleep 2
    if process_running "${pid}"; then
        pkill -TERM -P "${pid}" 2>/dev/null || true
        sleep 1
    fi
    if process_running "${pid}"; then
        kill -9 "${pid}" 2>/dev/null || true
    fi
}

echo "============================================================"
echo " BlueprintPipeline GCS-Synced Runner"
echo "============================================================"
echo "Scene ID:          ${SCENE_ID}"
echo "Bucket:            ${BUCKET}"
echo "Object:            ${OBJECT_NAME}"
echo "Object generation: ${OBJECT_GENERATION}"
echo "Steps:             ${STEPS}"
echo "Scene dir:         ${SCENE_DIR}"
echo "Log file:          ${LOG_FILE}"
echo "Started:           $(timestamp_utc)"
echo "============================================================"

# ─── Generation guard / latest-wins arbitration ───────────────────────────────
lock_state
read_state
if [ -n "${CURRENT_GENERATION}" ]; then
    cmp="$(compare_generation "${OBJECT_GENERATION}" "${CURRENT_GENERATION}")"
    if [ "${cmp}" -lt 0 ]; then
        echo "[GCS-Runner] Stale event ignored (incoming generation ${OBJECT_GENERATION} < current ${CURRENT_GENERATION})."
        unlock_state
        exit 0
    fi
    if [ "${cmp}" -eq 0 ]; then
        echo "[GCS-Runner] Duplicate event ignored (generation ${OBJECT_GENERATION} already processed/running)."
        unlock_state
        exit 0
    fi
    stop_existing_pid "${CURRENT_PID}"
fi
write_state "${OBJECT_GENERATION}" "$$" "launching"
unlock_state

# ─── Create scene directory ───────────────────────────────────────────────────
mkdir -p "${SCENE_DIR}"

# ─── Run pipeline with GCS sync ───────────────────────────────────────────────
echo "[GCS-Runner] Starting pipeline with GCS sync..."
echo ""
PIPELINE_EXIT=0
if ! preflight_runner_dependencies | tee "${LOG_FILE}"; then
    PIPELINE_EXIT=1
    lock_state
    read_state
    if [ "${CURRENT_GENERATION}" = "${OBJECT_GENERATION}" ]; then
        write_state "${OBJECT_GENERATION}" "" "failed"
    fi
    unlock_state
else
    set +e
    python3 "${SCRIPT_DIR}/tools/run_local_pipeline.py" \
        --scene-dir "${SCENE_DIR}" \
        --steps "${STEPS}" \
        --gcs-bucket "${BUCKET}" \
        --gcs-download-inputs \
        --gcs-upload-outputs \
        --gcs-input-object "${OBJECT_NAME}" \
        --gcs-input-generation "${OBJECT_GENERATION}" \
        --gcs-upload-concurrency "${GCS_UPLOAD_CONCURRENCY:-4}" \
        --fail-fast \
        > >(tee "${LOG_FILE}") 2>&1 &
    PIPE_PID=$!

    lock_state
    write_state "${OBJECT_GENERATION}" "${PIPE_PID}" "running"
    unlock_state

    wait "${PIPE_PID}"
    PIPELINE_EXIT=$?
    set -e

    lock_state
    read_state
    if [ "${CURRENT_GENERATION}" = "${OBJECT_GENERATION}" ]; then
        if [ "${PIPELINE_EXIT}" -eq 0 ]; then
            write_state "${OBJECT_GENERATION}" "" "completed"
        else
            write_state "${OBJECT_GENERATION}" "" "failed"
        fi
    fi
    unlock_state
fi

echo ""
echo "============================================================"
echo " Pipeline finished with exit code: ${PIPELINE_EXIT}"
echo " Completed: $(timestamp_utc)"
echo "============================================================"

# ─── Upload log file ──────────────────────────────────────────────────────────
if command -v gsutil >/dev/null 2>&1; then
    echo "[GCS-Runner] Uploading log file..."
    gsutil -q cp "${LOG_FILE}" \
        "gs://${BUCKET}/scenes/${SCENE_ID}/logs/pipeline_$(date -u +%Y%m%dT%H%M%S).log" \
        2>/dev/null || echo "[GCS-Runner] Log upload failed (non-critical)"
fi

exit "${PIPELINE_EXIT}"
