#!/usr/bin/env bash
set -euo pipefail

log() { echo "[sage-run $(date -u +%FT%TZ)] $*"; }

WORKSPACE=${WORKSPACE:-/workspace}
SAGE_DIR="${WORKSPACE}/SAGE"
MINICONDA_DIR="${WORKSPACE}/miniconda3"
SECRETS_ENV_PATH="${WORKSPACE}/.sage_runpod_secrets.env"

ROOM_DESC="${ROOM_DESC:-${1:-}}"
if [[ -z "${ROOM_DESC}" ]]; then
  log "ERROR: ROOM_DESC is required (pass as arg 1 or env ROOM_DESC)."
  exit 2
fi

SERVER_PATH="${SERVER_PATH:-../server/layout_wo_robot.py}"
INPUT_IMAGE_PATH="${INPUT_IMAGE_PATH:-}"

if [[ -f "${SECRETS_ENV_PATH}" ]]; then
  # shellcheck disable=SC1090
  source "${SECRETS_ENV_PATH}"
fi

if [[ ! -x "${MINICONDA_DIR}/bin/conda" ]]; then
  log "ERROR: Miniconda not found at ${MINICONDA_DIR}. Run bootstrap_sage_isaacsim_pod.sh first."
  exit 3
fi

# shellcheck disable=SC1091
source "${MINICONDA_DIR}/etc/profile.d/conda.sh"
conda activate sage

cd "${SAGE_DIR}/client"

log "Running generation"
log "  room_desc: ${ROOM_DESC}"
log "  server_path: ${SERVER_PATH}"
if [[ -n "${INPUT_IMAGE_PATH}" ]]; then
  log "  input_image: ${INPUT_IMAGE_PATH}"
  python client_generation_room_desc.py \
    --room_desc "${ROOM_DESC}" \
    --input_images "${INPUT_IMAGE_PATH}" \
    --server_paths "${SERVER_PATH}"
else
  python client_generation_room_desc.py \
    --room_desc "${ROOM_DESC}" \
    --server_paths "${SERVER_PATH}"
fi

