#!/bin/bash
# =============================================================================
# Setup EventArc trigger for source-orchestrator (text-first scene requests)
# =============================================================================
#
# Usage:
#   ./setup-source-orchestrator-trigger.sh <project_id> <bucket_name> [workflow_region]
#
# Trigger target:
#   scenes/{scene_id}/prompts/scene_request.json
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ID=${1:-$(gcloud config get-value project)}
BUCKET=${2:-"${PROJECT_ID}.appspot.com"}
WORKFLOW_REGION=${3:-"us-central1"}

# Eventarc trigger location must match bucket location (appspot buckets are multi-region "US")
TRIGGER_LOCATION="us"

TRIGGER_NAME="scene-request-source-orchestrator-trigger"
WORKFLOW_NAME="source-orchestrator"
WORKFLOW_SA="workflow-invoker"

echo -e "${GREEN}=== Source Orchestrator Trigger Setup ===${NC}"
echo "Project:           ${PROJECT_ID}"
echo "Bucket:            ${BUCKET}"
echo "Trigger location:  ${TRIGGER_LOCATION}"
echo "Workflow region:   ${WORKFLOW_REGION}"
echo ""

echo -e "${BLUE}Step 1: Enabling required APIs...${NC}"
gcloud services enable \
    eventarc.googleapis.com \
    workflows.googleapis.com \
    workflowexecutions.googleapis.com \
    run.googleapis.com \
    storage.googleapis.com \
    --project=${PROJECT_ID}
echo -e "${GREEN}APIs enabled${NC}"
echo ""

echo -e "${BLUE}Step 2: Setting up service account...${NC}"
SA_EMAIL="${WORKFLOW_SA}@${PROJECT_ID}.iam.gserviceaccount.com"
PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"

if ! gcloud iam service-accounts describe ${SA_EMAIL} --project=${PROJECT_ID} 2>/dev/null; then
    echo -e "${YELLOW}Creating service account ${WORKFLOW_SA}...${NC}"
    gcloud iam service-accounts create ${WORKFLOW_SA} \
        --display-name="Workflow Invoker SA" \
        --project=${PROJECT_ID}
    sleep 5
else
    echo "Service account ${WORKFLOW_SA} already exists"
fi

echo "Granting permissions..."
for ROLE in \
    "roles/workflows.invoker" \
    "roles/run.invoker" \
    "roles/storage.objectAdmin" \
    "roles/logging.logWriter" \
    "roles/eventarc.eventReceiver"; do
    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
        --member="serviceAccount:${SA_EMAIL}" \
        --role="${ROLE}" \
        --quiet 2>&1 | tail -1
done

GCS_SA="service-${PROJECT_NUMBER}@gs-project-accounts.iam.gserviceaccount.com"
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${GCS_SA}" \
    --role="roles/pubsub.publisher" \
    --quiet 2>&1 | tail -1

echo -e "${GREEN}Permissions granted${NC}"
echo ""

echo -e "${BLUE}Step 3: Deploying source orchestrator workflow...${NC}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
gcloud workflows deploy ${WORKFLOW_NAME} \
    --location=${WORKFLOW_REGION} \
    --source="${SCRIPT_DIR}/source-orchestrator.yaml" \
    --service-account=${SA_EMAIL} \
    --project=${PROJECT_ID}
echo -e "${GREEN}Workflow deployed${NC}"
echo ""

echo -e "${BLUE}Step 4: Recreating trigger if needed...${NC}"
if gcloud eventarc triggers describe ${TRIGGER_NAME} --location=${TRIGGER_LOCATION} --project=${PROJECT_ID} 2>/dev/null; then
    echo -e "${YELLOW}Trigger ${TRIGGER_NAME} exists; deleting and recreating...${NC}"
    gcloud eventarc triggers delete ${TRIGGER_NAME} \
        --location=${TRIGGER_LOCATION} \
        --project=${PROJECT_ID} \
        --quiet
    sleep 5
fi

echo -e "${BLUE}Step 5: Creating EventArc trigger...${NC}"
gcloud eventarc triggers create ${TRIGGER_NAME} \
    --location=${TRIGGER_LOCATION} \
    --project=${PROJECT_ID} \
    --service-account="${SA_EMAIL}" \
    --destination-workflow=${WORKFLOW_NAME} \
    --destination-workflow-location=${WORKFLOW_REGION} \
    --event-filters="type=google.cloud.storage.object.v1.finalized" \
    --event-filters="bucket=${BUCKET}" \
    --event-data-content-type="application/json"
echo -e "${GREEN}Trigger created${NC}"
echo ""

echo -e "${BLUE}Step 6: Verifying trigger...${NC}"
gcloud eventarc triggers describe ${TRIGGER_NAME} \
    --location=${TRIGGER_LOCATION} \
    --project=${PROJECT_ID} \
    --format="table(displayName, destination.workflow, eventFilters)"
echo ""

echo -e "${GREEN}=== Setup Complete ===${NC}"
echo "Workflow: ${WORKFLOW_NAME}"
echo "Trigger:  ${TRIGGER_NAME}"
echo ""
echo "The workflow triggers on bucket object finalize events and filters internally for:"
echo "  scenes/{scene_id}/prompts/scene_request.json"
echo ""
echo "To test:"
echo "  gsutil cp request.json gs://${BUCKET}/scenes/test_scene/prompts/scene_request.json"
