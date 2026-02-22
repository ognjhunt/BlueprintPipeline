#!/bin/bash
# Setup EventArc trigger for DWM preparation pipeline
#
# This script creates an EventArc trigger that invokes the dwm-preparation-pipeline
# workflow whenever a .stage1_complete marker file is uploaded to GCS.
#
# Usage:
#   ./setup_eventarc_trigger.sh <project_id> <bucket_name>
#
# Prerequisites:
#   - gcloud CLI authenticated
#   - dwm-preparation-pipeline workflow deployed
#   - Cloud Run, Workflows, and EventArc APIs enabled

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
PROJECT_ID=${1:-$(gcloud config get-value project)}
BUCKET=${2:-"${PROJECT_ID}-blueprint-scenes"}
REGION=${3:-"us-central1"}

# Derived values
TRIGGER_NAME="dwm-preparation-trigger"
WORKFLOW_NAME="dwm-preparation-pipeline"
WORKFLOW_SA="workflow-invoker"

echo -e "${GREEN}=== DWM EventArc Trigger Setup ===${NC}"
echo "Project: ${PROJECT_ID}"
echo "Bucket: ${BUCKET}"
echo "Region: ${REGION}"
echo ""

# Check if trigger already exists
if gcloud eventarc triggers describe ${TRIGGER_NAME} --location=${REGION} --project=${PROJECT_ID} 2>/dev/null; then
    echo -e "${YELLOW}Trigger ${TRIGGER_NAME} already exists. Deleting and recreating...${NC}"
    gcloud eventarc triggers delete ${TRIGGER_NAME} \
        --location=${REGION} \
        --project=${PROJECT_ID} \
        --quiet
fi

# Create service account if it doesn't exist
SA_EMAIL="${WORKFLOW_SA}@${PROJECT_ID}.iam.gserviceaccount.com"
if ! gcloud iam service-accounts describe ${SA_EMAIL} --project=${PROJECT_ID} 2>/dev/null; then
    echo -e "${YELLOW}Creating service account ${WORKFLOW_SA}...${NC}"
    gcloud iam service-accounts create ${WORKFLOW_SA} \
        --display-name="Workflow Invoker SA" \
        --project=${PROJECT_ID}

    # Grant necessary permissions
    echo "Granting permissions..."

    # Allow invoking workflows
    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
        --member="serviceAccount:${SA_EMAIL}" \
        --role="roles/workflows.invoker" \
        --quiet

    # Allow running Cloud Run jobs
    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
        --member="serviceAccount:${SA_EMAIL}" \
        --role="roles/run.invoker" \
        --quiet

    # Allow reading/writing GCS
    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
        --member="serviceAccount:${SA_EMAIL}" \
        --role="roles/storage.objectAdmin" \
        --quiet
fi

# Create EventArc trigger
echo -e "${GREEN}Creating EventArc trigger...${NC}"
gcloud eventarc triggers create ${TRIGGER_NAME} \
    --location=${REGION} \
    --project=${PROJECT_ID} \
    --service-account="${SA_EMAIL}" \
    --destination-workflow=${WORKFLOW_NAME} \
    --destination-workflow-location=${REGION} \
    --event-filters="type=google.cloud.storage.object.v1.finalized" \
    --event-filters="bucket=${BUCKET}" \
    --event-data-content-type="application/json"

echo ""
echo -e "${GREEN}=== Setup Complete ===${NC}"
echo ""
echo "Trigger created: ${TRIGGER_NAME}"
echo "Workflow: ${WORKFLOW_NAME}"
echo "Bucket: ${BUCKET}"
echo ""
echo "The pipeline will now trigger when .stage1_complete markers are uploaded to:"
echo "  gs://${BUCKET}/scenes/{scene_id}/assets/.stage1_complete"
echo ""
echo "To test manually:"
echo "  echo 'test' | gsutil cp - gs://${BUCKET}/scenes/test_scene/assets/.stage1_complete"
echo ""
echo "To view trigger:"
echo "  gcloud eventarc triggers describe ${TRIGGER_NAME} --location=${REGION}"
