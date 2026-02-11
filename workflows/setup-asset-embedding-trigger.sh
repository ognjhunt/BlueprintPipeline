#!/bin/bash
# =============================================================================
# Setup EventArc trigger for asset-embedding-pipeline
# =============================================================================
#
# Usage:
#   ./setup-asset-embedding-trigger.sh <project_id> <bucket_name> [workflow_region]
#

set -e

PROJECT_ID=${1:-$(gcloud config get-value project)}
BUCKET=${2:-"${PROJECT_ID}-blueprint-scenes"}
WORKFLOW_REGION=${3:-"us-central1"}
TRIGGER_LOCATION="us"

WORKFLOW_NAME="asset-embedding-pipeline"
TRIGGER_NAME="asset-embedding-queue-trigger"
WORKFLOW_SA="workflow-invoker"

ASSET_EMBEDDING_JOB_NAME=${ASSET_EMBEDDING_JOB_NAME:-"asset-embedding-job"}
TEXT_ASSET_EMBEDDING_QUEUE_PREFIX=${TEXT_ASSET_EMBEDDING_QUEUE_PREFIX:-"automation/asset_embedding/queue"}
TEXT_ASSET_EMBEDDING_PROCESSED_PREFIX=${TEXT_ASSET_EMBEDDING_PROCESSED_PREFIX:-"automation/asset_embedding/processed"}
TEXT_ASSET_EMBEDDING_FAILED_PREFIX=${TEXT_ASSET_EMBEDDING_FAILED_PREFIX:-"automation/asset_embedding/failed"}
TEXT_ASSET_EMBEDDING_MODEL=${TEXT_ASSET_EMBEDDING_MODEL:-"text-embedding-3-small"}
TEXT_ASSET_EMBEDDING_BACKEND=${TEXT_ASSET_EMBEDDING_BACKEND:-"openai"}
TEXT_ASSET_ANN_NAMESPACE=${TEXT_ASSET_ANN_NAMESPACE:-"assets-v1"}
VECTOR_STORE_PROVIDER=${VECTOR_STORE_PROVIDER:-"vertex"}
VECTOR_STORE_PROJECT_ID=${VECTOR_STORE_PROJECT_ID:-"${PROJECT_ID}"}
VECTOR_STORE_LOCATION=${VECTOR_STORE_LOCATION:-"${WORKFLOW_REGION}"}
VECTOR_STORE_NAMESPACE=${VECTOR_STORE_NAMESPACE:-"${TEXT_ASSET_ANN_NAMESPACE}"}
VECTOR_STORE_DIMENSION=${VECTOR_STORE_DIMENSION:-"1536"}
VERTEX_INDEX_ENDPOINT=${VERTEX_INDEX_ENDPOINT:-""}
VERTEX_DEPLOYED_INDEX_ID=${VERTEX_DEPLOYED_INDEX_ID:-""}
VERTEX_INDEX_NAME=${VERTEX_INDEX_NAME:-""}
OPENAI_API_KEY_SECRET=${OPENAI_API_KEY_SECRET:-""}

echo "=== Asset Embedding Trigger Setup ==="
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
  --source="${SCRIPT_DIR}/asset-embedding-pipeline.yaml" \
  --set-env-vars="ASSET_EMBEDDING_JOB_NAME=${ASSET_EMBEDDING_JOB_NAME},TEXT_ASSET_EMBEDDING_QUEUE_PREFIX=${TEXT_ASSET_EMBEDDING_QUEUE_PREFIX}" \
  --service-account="${SA_EMAIL}" \
  --project="${PROJECT_ID}"

if gcloud run jobs describe "${ASSET_EMBEDDING_JOB_NAME}" --region="${WORKFLOW_REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  JOB_SERVICE_ACCOUNT="$(gcloud run jobs describe "${ASSET_EMBEDDING_JOB_NAME}" --region="${WORKFLOW_REGION}" --project="${PROJECT_ID}" --format='value(template.template.serviceAccount)')"
  JOB_ENV_VARS="TEXT_ASSET_EMBEDDING_QUEUE_PREFIX=${TEXT_ASSET_EMBEDDING_QUEUE_PREFIX},TEXT_ASSET_EMBEDDING_PROCESSED_PREFIX=${TEXT_ASSET_EMBEDDING_PROCESSED_PREFIX},TEXT_ASSET_EMBEDDING_FAILED_PREFIX=${TEXT_ASSET_EMBEDDING_FAILED_PREFIX},TEXT_ASSET_EMBEDDING_MODEL=${TEXT_ASSET_EMBEDDING_MODEL},TEXT_ASSET_EMBEDDING_BACKEND=${TEXT_ASSET_EMBEDDING_BACKEND},TEXT_ASSET_ANN_NAMESPACE=${TEXT_ASSET_ANN_NAMESPACE},VECTOR_STORE_PROVIDER=${VECTOR_STORE_PROVIDER},VECTOR_STORE_PROJECT_ID=${VECTOR_STORE_PROJECT_ID},VECTOR_STORE_LOCATION=${VECTOR_STORE_LOCATION},VECTOR_STORE_NAMESPACE=${VECTOR_STORE_NAMESPACE},VECTOR_STORE_DIMENSION=${VECTOR_STORE_DIMENSION},VERTEX_INDEX_ENDPOINT=${VERTEX_INDEX_ENDPOINT},VERTEX_DEPLOYED_INDEX_ID=${VERTEX_DEPLOYED_INDEX_ID},VERTEX_INDEX_NAME=${VERTEX_INDEX_NAME}"
  gcloud run jobs update "${ASSET_EMBEDDING_JOB_NAME}" \
    --region="${WORKFLOW_REGION}" \
    --project="${PROJECT_ID}" \
    --update-env-vars="${JOB_ENV_VARS}" >/dev/null

  if [ -n "${OPENAI_API_KEY_SECRET}" ]; then
    gcloud run jobs update "${ASSET_EMBEDDING_JOB_NAME}" \
      --region="${WORKFLOW_REGION}" \
      --project="${PROJECT_ID}" \
      --update-secrets="OPENAI_API_KEY=${OPENAI_API_KEY_SECRET}:latest" >/dev/null
    if [ -n "${JOB_SERVICE_ACCOUNT}" ] && gcloud secrets describe "${OPENAI_API_KEY_SECRET}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
      gcloud secrets add-iam-policy-binding "${OPENAI_API_KEY_SECRET}" \
        --project="${PROJECT_ID}" \
        --member="serviceAccount:${JOB_SERVICE_ACCOUNT}" \
        --role="roles/secretmanager.secretAccessor" \
        --quiet >/dev/null
    fi
  else
    echo "WARNING: OPENAI_API_KEY_SECRET not set; embedding job must already have OPENAI_API_KEY configured."
  fi
else
  echo "WARNING: Cloud Run job ${ASSET_EMBEDDING_JOB_NAME} not found; skipped runtime env/secret configuration."
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

