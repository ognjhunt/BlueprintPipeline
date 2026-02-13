#!/usr/bin/env bash
set -euo pipefail

log() { echo "[trellis-bootstrap $(date -u +%FT%TZ)] $*"; }

WORKSPACE=${WORKSPACE:-/workspace}
SAGE_DIR="${WORKSPACE}/SAGE"
LOG_PATH="${WORKSPACE}/trellis_server.log"
SECRETS_ENV_PATH="${WORKSPACE}/.sage_runpod_secrets.env"

if [[ -f "${SECRETS_ENV_PATH}" ]]; then
  # shellcheck disable=SC1090
  source "${SECRETS_ENV_PATH}"
  log "Loaded secrets env: ${SECRETS_ENV_PATH}"
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  log "ERROR: HF_TOKEN is required (HuggingFace token for TRELLIS checkpoint download)."
  exit 2
fi

log "Configuring HuggingFace caches under ${WORKSPACE}"
export HF_HOME="${HF_HOME:-${WORKSPACE}/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

log "Preflight: GPU visibility"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  log "ERROR: nvidia-smi not found (GPU not available in this pod)."
  exit 3
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n 1 || true

log "Ensuring base packages"
apt-get update -qq
apt-get install -y -qq git curl ca-certificates >/dev/null

if [[ ! -d "${SAGE_DIR}/.git" ]]; then
  log "Cloning NVlabs/sage into ${SAGE_DIR}"
  git clone --depth 1 https://github.com/NVlabs/sage.git "${SAGE_DIR}"
else
  log "SAGE repo already present"
fi

log "Checking if TRELLIS server is already up"
if curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1; then
  log "TRELLIS already healthy on :8080"
  exit 0
fi

log "Starting TRELLIS server in background (this can take a long time on first run)"
cd "${SAGE_DIR}/server"

# The upstream script blocks in the foreground once the server starts. Run it under nohup.
nohup bash start_trellis_server.sh >"${LOG_PATH}" 2>&1 &

log "Waiting for http://127.0.0.1:8080/health (up to 60 minutes)"
deadline=$(( $(date +%s) + 3600 ))
while [[ "$(date +%s)" -lt "${deadline}" ]]; do
  if curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1; then
    log "TRELLIS is healthy"
    exit 0
  fi
  sleep 10
done

log "ERROR: TRELLIS did not become healthy within timeout."
log "Last 80 log lines from ${LOG_PATH}:"
tail -n 80 "${LOG_PATH}" || true
exit 4
