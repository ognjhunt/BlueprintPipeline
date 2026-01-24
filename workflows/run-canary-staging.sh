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
REPO_ROOT="$(git rev-parse --show-toplevel)"
VALIDATION_SCRIPT="${VALIDATION_SCRIPT:-${REPO_ROOT}/workflows/validate-geniesim-export-artifacts.py}"
SCHEMA_DIR="${SCHEMA_DIR:-${REPO_ROOT}/fixtures/contracts}"
VALIDATION_JOB_NAME="${VALIDATION_JOB_NAME:-canary-validation}"

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

write_failure_marker() {
  local scene_id="$1"
  local label="$2"
  local validation_log="$3"

  python - <<PY
from pathlib import Path

from tools.workflow.failure_markers import FailureMarkerWriter

scene_id = "${scene_id}"
label = "${label}"
validation_log = Path("${validation_log}")
message = validation_log.read_text().strip() if validation_log.exists() else "Canary validation failed."

writer = FailureMarkerWriter(
    bucket="${BUCKET}",
    scene_id=scene_id,
    job_name="${VALIDATION_JOB_NAME}",
)

writer.write_failure(
    exception=RuntimeError(message),
    failed_step="validate_geniesim_exports",
    input_params={
        "scene_id": scene_id,
        "label": label,
        "bucket": "${BUCKET}",
        "schema_dir": "${SCHEMA_DIR}",
    },
    partial_results={"validation_log": str(validation_log)},
    recommendations=[
        "Inspect the validation log for missing artifacts or schema errors.",
        "Re-run genie-sim-export for the scene after fixing export issues.",
    ],
    error_code="CANARY_VALIDATION_FAILED",
)
PY
}

validate_geniesim_artifacts() {
  local label="$1"
  local scene_id="$2"
  local artifacts_dir="${ARTIFACT_ROOT}/${label}-geniesim"
  local validation_log="${ARTIFACT_ROOT}/${label}-validation.log"
  local validation_report="${ARTIFACT_ROOT}/${label}-validation.json"
  local gcs_prefix="gs://${BUCKET}/scenes/${scene_id}/geniesim"

  mkdir -p "${artifacts_dir}"
  : > "${validation_log}"

  echo "Downloading Genie Sim artifacts for ${label} (${scene_id})."
  if ! gsutil cp "${gcs_prefix}/scene_graph.json" "${artifacts_dir}/" 2>>"${validation_log}"; then
    echo "Missing scene_graph.json for ${scene_id}." >> "${validation_log}"
    write_failure_marker "${scene_id}" "${label}" "${validation_log}"
    return 1
  fi
  if ! gsutil cp "${gcs_prefix}/asset_index.json" "${artifacts_dir}/" 2>>"${validation_log}"; then
    echo "Missing asset_index.json for ${scene_id}." >> "${validation_log}"
    write_failure_marker "${scene_id}" "${label}" "${validation_log}"
    return 1
  fi
  if ! gsutil cp "${gcs_prefix}/task_config.json" "${artifacts_dir}/" 2>>"${validation_log}"; then
    echo "Missing task_config.json for ${scene_id}." >> "${validation_log}"
    write_failure_marker "${scene_id}" "${label}" "${validation_log}"
    return 1
  fi

  if ! python "${VALIDATION_SCRIPT}" \
    --scene-graph "${artifacts_dir}/scene_graph.json" \
    --asset-index "${artifacts_dir}/asset_index.json" \
    --task-config "${artifacts_dir}/task_config.json" \
    --schema-dir "${SCHEMA_DIR}" \
    > "${validation_report}" 2>>"${validation_log}"; then
    echo "Canary artifact validation failed for ${label}; see ${validation_log}." >&2
    write_failure_marker "${scene_id}" "${label}" "${validation_log}"
    return 1
  fi

  echo "Canary artifact validation passed for ${label}. Report: ${validation_report}"
}

echo "Running stable execution (expected stable image tag)."
run_execution "stable" "${STABLE_SCENE_ID}" "false" "${CANARY_ROLLBACK_MARKER}"

echo "Running canary execution (expected canary image tag)."
run_execution "canary" "${CANARY_SCENE_ID}" "true" "${CANARY_ROLLBACK_MARKER}"

echo "Validating canary export artifacts."
validate_geniesim_artifacts "canary" "${CANARY_SCENE_ID}"

if [[ -n "${CANARY_ROLLBACK_MARKER}" ]]; then
  echo "Running rollback marker execution (expected rollback skip)."
  run_execution "rollback" "${CANARY_SCENE_ID}" "true" "${CANARY_ROLLBACK_MARKER}"
else
  echo "CANARY_ROLLBACK_MARKER not set; skipping rollback marker validation run."
fi

echo "Canary staging validation complete. Review artifacts under ${ARTIFACT_ROOT}."
