#!/usr/bin/env bash
# =============================================================================
# Start Interactive Asset Backend Services (localhost)
#
# Starts PhysX-Anything and Infinigen as local HTTP services on the pod.
# These are consumed by run_interactive_assets.py via PHYSX_ANYTHING_ENDPOINT
# and INFINIGEN_ENDPOINT environment variables.
#
# Usage:
#   bash start_interactive_backends.sh          # start both
#   bash start_interactive_backends.sh physx    # start PhysX-Anything only
#   bash start_interactive_backends.sh infinigen # start Infinigen only
#   bash start_interactive_backends.sh stop     # stop all
#
# Ports:
#   PhysX-Anything:  localhost:8083
#   Infinigen:       localhost:8084
# =============================================================================
set -euo pipefail

log() { echo "[backends $(date -u +%FT%TZ)] $*"; }

WORKSPACE="${WORKSPACE:-/workspace}"
CONDA_DIR="${WORKSPACE}/miniconda3"
SAGE_ENV="sage"

PHYSX_PORT=8083
INFINIGEN_PORT=8084

PHYSX_ANYTHING_DIR="${WORKSPACE}/PhysX-Anything"
INFINIGEN_DIR="${WORKSPACE}/infinigen"

ACTION="${1:-all}"  # all | physx | infinigen | stop | status

# Activate conda
if [[ -f "${CONDA_DIR}/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${CONDA_DIR}/etc/profile.d/conda.sh"
    conda activate "${SAGE_ENV}" 2>/dev/null || true
fi

PYTHON="$(which python 2>/dev/null || echo python3)"

stop_backend() {
    local name="$1" pidfile="$2"
    if [[ -f "${pidfile}" ]]; then
        local pid
        pid="$(cat "${pidfile}" 2>/dev/null || true)"
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            log "Stopping ${name} (pid=${pid})..."
            kill "${pid}" 2>/dev/null || true
            sleep 2
            if kill -0 "${pid}" 2>/dev/null; then
                kill -9 "${pid}" 2>/dev/null || true
            fi
            log "${name} stopped."
        fi
        rm -f "${pidfile}"
    fi
}

status_backend() {
    local name="$1" port="$2" pidfile="$3"
    local pid=""
    if [[ -f "${pidfile}" ]]; then
        pid="$(cat "${pidfile}" 2>/dev/null || true)"
    fi

    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        local health
        health="$(curl -sf "http://localhost:${port}/" 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("status","?"), "ready" if d.get("ready") else "not ready")' 2>/dev/null || echo "no response")"
        log "  ${name}: RUNNING (pid=${pid}, port=${port}, health=${health})"
    else
        log "  ${name}: STOPPED"
    fi
}

start_physx() {
    local svc="${WORKSPACE}/physx_anything_service.py"
    if [[ ! -f "${svc}" ]]; then
        # Fall back to repo copy
        svc="${WORKSPACE}/BlueprintPipeline/physx-anything-service/physx_anything_service.py"
    fi
    if [[ ! -f "${svc}" ]]; then
        log "ERROR: physx_anything_service.py not found"
        return 1
    fi
    if [[ ! -d "${PHYSX_ANYTHING_DIR}/pretrain/vlm" ]]; then
        log "ERROR: PhysX-Anything weights not found at ${PHYSX_ANYTHING_DIR}/pretrain/vlm"
        log "  Run install_interactive_backends.sh first."
        return 1
    fi

    stop_backend "PhysX-Anything" /tmp/physx_anything.pid

    log "Starting PhysX-Anything on :${PHYSX_PORT}..."
    PHYSX_ANYTHING_ROOT="${PHYSX_ANYTHING_DIR}" \
    PHYSX_ANYTHING_CKPT="${PHYSX_ANYTHING_DIR}/pretrain/vlm" \
    PORT="${PHYSX_PORT}" \
    nohup "${PYTHON}" "${svc}" > /tmp/physx_anything_service.log 2>&1 &
    echo $! > /tmp/physx_anything.pid
    log "PhysX-Anything PID: $(cat /tmp/physx_anything.pid)"
}

start_infinigen() {
    local svc="${WORKSPACE}/infinigen_service.py"
    if [[ ! -f "${svc}" ]]; then
        svc="${WORKSPACE}/BlueprintPipeline/infinigen-service/infinigen_service.py"
    fi
    if [[ ! -f "${svc}" ]]; then
        log "ERROR: infinigen_service.py not found"
        return 1
    fi
    if [[ ! -f "${INFINIGEN_DIR}/scripts/spawn_asset.py" ]]; then
        log "ERROR: Infinigen not found at ${INFINIGEN_DIR}"
        log "  Run install_interactive_backends.sh first."
        return 1
    fi

    # Infinigen requires Python 3.11 (separate conda env)
    local INFINIGEN_PYTHON="${CONDA_DIR}/envs/infinigen/bin/python"
    if [[ ! -x "${INFINIGEN_PYTHON}" ]]; then
        log "WARNING: Infinigen conda env not found, falling back to ${PYTHON}"
        INFINIGEN_PYTHON="${PYTHON}"
    fi

    stop_backend "Infinigen" /tmp/infinigen.pid

    log "Starting Infinigen on :${INFINIGEN_PORT} (python: ${INFINIGEN_PYTHON})..."
    INFINIGEN_ROOT="${INFINIGEN_DIR}" \
    PORT="${INFINIGEN_PORT}" \
    nohup "${INFINIGEN_PYTHON}" "${svc}" > /tmp/infinigen_service.log 2>&1 &
    echo $! > /tmp/infinigen.pid
    log "Infinigen PID: $(cat /tmp/infinigen.pid)"
}

case "${ACTION}" in
    all)
        log "Starting all interactive backends..."
        start_physx
        start_infinigen
        log ""
        log "Backends started. Use these env vars with the pipeline:"
        log "  export ARTICULATION_BACKEND=auto"
        log "  export PHYSX_ANYTHING_ENABLED=true"
        log "  export PHYSX_ANYTHING_ENDPOINT=http://localhost:${PHYSX_PORT}"
        log "  export INFINIGEN_ENABLED=true"
        log "  export INFINIGEN_ENDPOINT=http://localhost:${INFINIGEN_PORT}"
        log ""
        log "Logs:"
        log "  PhysX-Anything: tail -f /tmp/physx_anything_service.log"
        log "  Infinigen:      tail -f /tmp/infinigen_service.log"
        ;;
    physx)
        start_physx
        ;;
    infinigen)
        start_infinigen
        ;;
    stop)
        log "Stopping all interactive backends..."
        stop_backend "PhysX-Anything" /tmp/physx_anything.pid
        stop_backend "Infinigen" /tmp/infinigen.pid
        log "All backends stopped."
        ;;
    status)
        log "Interactive backend status:"
        status_backend "PhysX-Anything" "${PHYSX_PORT}" /tmp/physx_anything.pid
        status_backend "Infinigen" "${INFINIGEN_PORT}" /tmp/infinigen.pid
        ;;
    *)
        echo "Usage: $0 [all|physx|infinigen|stop|status]"
        exit 1
        ;;
esac
