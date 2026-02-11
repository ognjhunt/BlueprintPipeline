#!/bin/bash
# =============================================================================
# Setup EventArc trigger for asset-replication-pipeline
# =============================================================================
#
# Usage:
#   ./setup-asset-replication-trigger.sh <project_id> <bucket_name> [workflow_region]
#

set -e

PROJECT_ID=${1:-$(gcloud config get-value project)}
BUCKET=${2:-"${PROJECT_ID}-blueprint-scenes"}
WORKFLOW_REGION=${3:-"us-central1"}
TRIGGER_LOCATION="us"

WORKFLOW_NAME="asset-replication-pipeline"
TRIGGER_NAME="asset-replication-queue-trigger"
WORKFLOW_SA="workflow-invoker"

ASSET_REPLICATION_JOB_NAME=${ASSET_REPLICATION_JOB_NAME:-"asset-replication-job"}
TEXT_ASSET_REPLICATION_QUEUE_PREFIX=${TEXT_ASSET_REPLICATION_QUEUE_PREFIX:-"automation/asset_replication/queue"}
B2_S3_ENDPOINT=${B2_S3_ENDPOINT:-""}
B2_BUCKET=${B2_BUCKET:-""}
B2_REGION=${B2_REGION:-"us-west-000"}
B2_KEY_ID_SECRET=${B2_KEY_ID_SECRET:-""}
B2_APPLICATION_KEY_SECRET=${B2_APPLICATION_KEY_SECRET:-""}

echo "=== Asset Replication Trigger Setup ==="
echo "Project: ${PROJECT_ID}"
echo "Bucket: ${BUCKET}"
echo "Region: ${WORKFLOW_REGION}"
echo ""

gcloud services enable \
  eventarc.googleapis.com \
  workflows.googleapis.com \
  workflowexecutions.googleapis.com \
  run.googleapis.com \
  storage.googleapis.com \
  --project="${PROJECT_ID}"

SA_EMAIL="${WORKFLOW_SA}@${PROJECT_ID}.iam.gserviceaccount.com"
if ! gcloud iam service-accounts describe "${SA_EMAIL}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud iam service-accounts create "${WORKFLOW_SA}" \
    --display-name="Workflow Invoker SA" \
    --project="${PROJECT_ID}"
  sleep 5
fi

for ROLE in "roles/workflows.invoker" "roles/run.invoker" "roles/storage.objectAdmin" "roles/eventarc.eventReceiver"; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="${ROLE}" \
    --quiet >/dev/null
done

PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
GCS_SA="service-${PROJECT_NUMBER}@gs-project-accounts.iam.gserviceaccount.com"
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${GCS_SA}" \
  --role="roles/pubsub.publisher" \
  --quiet >/dev/null

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
gcloud workflows deploy "${WORKFLOW_NAME}" \
  --location="${WORKFLOW_REGION}" \
  --source="${SCRIPT_DIR}/asset-replication-pipeline.yaml" \
  --set-env-vars="ASSET_REPLICATION_JOB_NAME=${ASSET_REPLICATION_JOB_NAME},TEXT_ASSET_REPLICATION_QUEUE_PREFIX=${TEXT_ASSET_REPLICATION_QUEUE_PREFIX}" \
  --service-account="${SA_EMAIL}" \
  --project="${PROJECT_ID}"

# Configure runtime env on the replication job itself. Credentials should come from
# Secret Manager bindings, not workflow env vars.
if gcloud run jobs describe "${ASSET_REPLICATION_JOB_NAME}" --region="${WORKFLOW_REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  JOB_SERVICE_ACCOUNT="$(gcloud run jobs describe "${ASSET_REPLICATION_JOB_NAME}" --region="${WORKFLOW_REGION}" --project="${PROJECT_ID}" --format='value(template.template.serviceAccount)')"
  JOB_ENV_VARS="B2_REGION=${B2_REGION}"
  if [ -n "${B2_S3_ENDPOINT}" ]; then
    JOB_ENV_VARS+=",B2_S3_ENDPOINT=${B2_S3_ENDPOINT}"
  fi
  if [ -n "${B2_BUCKET}" ]; then
    JOB_ENV_VARS+=",B2_BUCKET=${B2_BUCKET}"
  fi

  gcloud run jobs update "${ASSET_REPLICATION_JOB_NAME}" \
    --region="${WORKFLOW_REGION}" \
    --project="${PROJECT_ID}" \
    --update-env-vars="${JOB_ENV_VARS}" >/dev/null

  if [ -n "${B2_KEY_ID_SECRET}" ] || [ -n "${B2_APPLICATION_KEY_SECRET}" ]; then
    if [ -z "${B2_KEY_ID_SECRET}" ] || [ -z "${B2_APPLICATION_KEY_SECRET}" ]; then
      echo "WARNING: Both B2_KEY_ID_SECRET and B2_APPLICATION_KEY_SECRET are required to bind secrets."
    else
      gcloud run jobs update "${ASSET_REPLICATION_JOB_NAME}" \
        --region="${WORKFLOW_REGION}" \
        --project="${PROJECT_ID}" \
        --update-secrets="B2_KEY_ID=${B2_KEY_ID_SECRET}:latest,B2_APPLICATION_KEY=${B2_APPLICATION_KEY_SECRET}:latest" >/dev/null
      if [ -n "${JOB_SERVICE_ACCOUNT}" ]; then
        for SECRET_NAME in "${B2_KEY_ID_SECRET}" "${B2_APPLICATION_KEY_SECRET}"; do
          if gcloud secrets describe "${SECRET_NAME}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
            gcloud secrets add-iam-policy-binding "${SECRET_NAME}" \
              --project="${PROJECT_ID}" \
              --member="serviceAccount:${JOB_SERVICE_ACCOUNT}" \
              --role="roles/secretmanager.secretAccessor" \
              --quiet >/dev/null
          else
            echo "WARNING: Secret ${SECRET_NAME} not found in project ${PROJECT_ID}; skipping IAM binding."
          fi
        done
      fi
      echo "Bound B2 credentials from Secret Manager to ${ASSET_REPLICATION_JOB_NAME}."
    fi
  else
    echo "WARNING: B2 secret bindings not configured. Set B2_KEY_ID_SECRET and B2_APPLICATION_KEY_SECRET."
  fi
else
  echo "WARNING: Cloud Run job ${ASSET_REPLICATION_JOB_NAME} not found; skipped runtime env/secret configuration."
fi

if gcloud eventarc triggers describe "${TRIGGER_NAME}" --location="${TRIGGER_LOCATION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud eventarc triggers delete "${TRIGGER_NAME}" \
    --location="${TRIGGER_LOCATION}" \
    --project="${PROJECT_ID}" \
    --quiet
  sleep 5
fi

gcloud eventarc triggers create "${TRIGGER_NAME}" \
  --location="${TRIGGER_LOCATION}" \
  --project="${PROJECT_ID}" \
  --service-account="${SA_EMAIL}" \
  --destination-workflow="${WORKFLOW_NAME}" \
  --destination-workflow-location="${WORKFLOW_REGION}" \
  --event-filters="type=google.cloud.storage.object.v1.finalized" \
  --event-filters="bucket=${BUCKET}" \
  --event-data-content-type="application/json"

echo ""
echo "Setup complete."
echo "Trigger: ${TRIGGER_NAME}"
echo "Workflow: ${WORKFLOW_NAME}"
