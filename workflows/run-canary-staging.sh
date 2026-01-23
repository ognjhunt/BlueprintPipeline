#!/usr/bin/env bash
set -euo pipefail

WORKFLOW_NAME="genie-sim-export-pipeline"
WORKFLOW_REGION="${WORKFLOW_REGION:-us-central1}"
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
BUCKET="${BUCKET:-}"
CANARY_SCENE_ID="${CANARY_SCENE_ID:-}"
STABLE_SCENE_ID="${STABLE_SCENE_ID:-}"
CANARY_IMAGE_TAG="${CANARY_IMAGE_TAG:-isaacsim-canary}"
CANARY_PERCENT="${CANARY_PERCENT:-5}"
CANARY_RELEASE_CHANNEL="${CANARY_RELEASE_CHANNEL:-canary}"
CANARY_ROLLBACK_MARKER="${CANARY_ROLLBACK_MARKER:-}"
RUN_TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-workflows/artifacts/canary-validation/${RUN_TIMESTAMP}}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "PROJECT_ID is not set and no gcloud default project found." >&2
  exit 1
fi

if [[ -z "${BUCKET}" ]]; then
  echo "BUCKET is required (GCS bucket with scenes)." >&2
  exit 1
fi

if [[ -z "${CANARY_SCENE_ID}" || -z "${STABLE_SCENE_ID}" ]]; then
  echo "CANARY_SCENE_ID and STABLE_SCENE_ID are required." >&2
  exit 1
fi

mkdir -p "${ARTIFACT_ROOT}"

echo "Artifacts will be stored in ${ARTIFACT_ROOT}"

event_payload() {
  local scene_id="$1"
  local canary_enabled="$2"
  local rollback_marker="$3"

  cat <<PAYLOAD
{
  "data": {
    "bucket": "${BUCKET}",
    "name": "scenes/${scene_id}/variation_assets/.variation_pipeline_complete"
  },
  "canary_enabled": ${canary_enabled},
  "canary_percent": "${CANARY_PERCENT}",
  "canary_release_channel": "${CANARY_RELEASE_CHANNEL}",
  "canary_image_tag": "${CANARY_IMAGE_TAG}",
  "canary_rollback_marker": "${rollback_marker}"
}
PAYLOAD
}

run_execution() {
  local label="$1"
  local scene_id="$2"
  local canary_enabled="$3"
  local rollback_marker="$4"

  local payload
  payload="$(event_payload "${scene_id}" "${canary_enabled}" "${rollback_marker}")"

  echo "Triggering ${label} execution for scene ${scene_id} (canary_enabled=${canary_enabled})."
  local execution_json
  execution_json="$(gcloud workflows run "${WORKFLOW_NAME}" \
    --location="${WORKFLOW_REGION}" \
    --data="${payload}" \
    --format=json)"

  echo "${execution_json}" > "${ARTIFACT_ROOT}/${label}-execution.json"

  local execution_name
  execution_name="$(echo "${execution_json}" | jq -r '.name')"

  if [[ -z "${execution_name}" || "${execution_name}" == "null" ]]; then
    echo "Unable to determine execution name for ${label}." >&2
    exit 1
  fi

  echo "Execution name: ${execution_name}"

  local execution_id
  execution_id="${execution_name##*/}"

  gcloud workflows executions describe "${execution_id}" \
    --workflow="${WORKFLOW_NAME}" \
    --location="${WORKFLOW_REGION}" \
    --format=json > "${ARTIFACT_ROOT}/${label}-execution-describe.json"

  gcloud logging read \
    "resource.type=\"workflows.googleapis.com/Workflow\" AND resource.labels.workflow_id=\"${WORKFLOW_NAME}\" AND labels.execution_id=\"${execution_id}\"" \
    --project="${PROJECT_ID}" \
    --format=json > "${ARTIFACT_ROOT}/${label}-workflow-logs.json" || true

  echo "Stored workflow logs to ${ARTIFACT_ROOT}/${label}-workflow-logs.json"
}

echo "Running stable execution (expected stable image tag)."
run_execution "stable" "${STABLE_SCENE_ID}" "false" "${CANARY_ROLLBACK_MARKER}"

echo "Running canary execution (expected canary image tag)."
run_execution "canary" "${CANARY_SCENE_ID}" "true" "${CANARY_ROLLBACK_MARKER}"

if [[ -n "${CANARY_ROLLBACK_MARKER}" ]]; then
  echo "Running rollback marker execution (expected rollback skip)."
  run_execution "rollback" "${CANARY_SCENE_ID}" "true" "${CANARY_ROLLBACK_MARKER}"
else
  echo "CANARY_ROLLBACK_MARKER not set; skipping rollback marker validation run."
fi

echo "Canary staging validation complete. Review artifacts under ${ARTIFACT_ROOT}."
