#!/bin/bash
# =============================================================================
# Setup EventArc trigger for USD Assembly Pipeline
# =============================================================================
#
# This script creates an EventArc trigger that invokes the usd-assembly-pipeline
# workflow whenever a .stage1_complete marker file is uploaded to GCS.
#
# The pipeline converts Stage 1 text generation GLB output to USDZ, prepares sim-ready assets,
# and generates Replicator bundles and Isaac Lab task packages.
#
# Usage:
#   ./setup-usd-assembly-trigger.sh <project_id> <bucket_name>
#
# Prerequisites:
#   - gcloud CLI authenticated
#   - usd-assembly-pipeline workflow deployed
#   - Cloud Run, Workflows, and EventArc APIs enabled
#
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
PROJECT_ID=${1:-$(gcloud config get-value project)}
BUCKET=${2:-"${PROJECT_ID}-blueprint-scenes"}
REGION=${3:-"us-central1"}

# Derived values
TRIGGER_NAME="usd-assembly-trigger"
WORKFLOW_NAME="usd-assembly-pipeline"
WORKFLOW_SA="workflow-invoker"

echo -e "${GREEN}=== USD Assembly EventArc Trigger Setup ===${NC}"
echo "Project: ${PROJECT_ID}"
echo "Bucket: ${BUCKET}"
echo "Region: ${REGION}"
echo ""

# =============================================================================
# Step 1: Enable required APIs
# =============================================================================
echo -e "${BLUE}Step 1: Enabling required APIs...${NC}"

gcloud services enable \
    eventarc.googleapis.com \
    workflows.googleapis.com \
    run.googleapis.com \
    storage.googleapis.com \
    --project=${PROJECT_ID}

echo -e "${GREEN}APIs enabled${NC}"
echo ""

# =============================================================================
# Step 2: Create/verify service account
# =============================================================================
echo -e "${BLUE}Step 2: Setting up service account...${NC}"

SA_EMAIL="${WORKFLOW_SA}@${PROJECT_ID}.iam.gserviceaccount.com"

if ! gcloud iam service-accounts describe ${SA_EMAIL} --project=${PROJECT_ID} 2>/dev/null; then
    echo -e "${YELLOW}Creating service account ${WORKFLOW_SA}...${NC}"
    gcloud iam service-accounts create ${WORKFLOW_SA} \
        --display-name="Workflow Invoker SA" \
        --project=${PROJECT_ID}

    sleep 5
else
    echo "Service account ${WORKFLOW_SA} already exists"
fi

# Grant necessary permissions
echo "Granting permissions..."

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/workflows.invoker" \
    --quiet

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/run.invoker" \
    --quiet

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/storage.objectAdmin" \
    --quiet

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/logging.logWriter" \
    --quiet

echo -e "${GREEN}Permissions granted${NC}"
echo ""

# =============================================================================
# Step 3: Delete existing trigger if present
# =============================================================================
echo -e "${BLUE}Step 3: Checking for existing trigger...${NC}"

if gcloud eventarc triggers describe ${TRIGGER_NAME} --location=${REGION} --project=${PROJECT_ID} 2>/dev/null; then
    echo -e "${YELLOW}Trigger ${TRIGGER_NAME} already exists. Deleting and recreating...${NC}"
    gcloud eventarc triggers delete ${TRIGGER_NAME} \
        --location=${REGION} \
        --project=${PROJECT_ID} \
        --quiet

    sleep 5
fi

# =============================================================================
# Step 4: Verify workflow exists
# =============================================================================
echo -e "${BLUE}Step 4: Verifying workflow exists...${NC}"

if ! gcloud workflows describe ${WORKFLOW_NAME} --location=${REGION} --project=${PROJECT_ID} 2>/dev/null; then
    echo -e "${YELLOW}Workflow ${WORKFLOW_NAME} not found. Deploying...${NC}"
    gcloud workflows deploy ${WORKFLOW_NAME} \
        --location=${REGION} \
        --source=usd-assembly-pipeline.yaml \
        --description="USD assembly pipeline (triggered by .stage1_complete)" \
        --service-account=${SA_EMAIL} \
        --project=${PROJECT_ID}
else
    echo "Workflow ${WORKFLOW_NAME} exists"
fi
echo ""

# =============================================================================
# Step 5: Create EventArc trigger
# =============================================================================
echo -e "${BLUE}Step 5: Creating EventArc trigger...${NC}"

gcloud eventarc triggers create ${TRIGGER_NAME} \
    --location=${REGION} \
    --project=${PROJECT_ID} \
    --service-account="${SA_EMAIL}" \
    --destination-workflow=${WORKFLOW_NAME} \
    --destination-workflow-location=${REGION} \
    --event-filters="type=google.cloud.storage.object.v1.finalized" \
    --event-filters="bucket=${BUCKET}" \
    --event-filters="name=^scenes/.+/assets/\\.stage1_complete$" \
    --event-data-content-type="application/json"

echo -e "${GREEN}EventArc trigger created${NC}"
echo ""

# =============================================================================
# Step 6: Verify setup
# =============================================================================
echo -e "${BLUE}Step 6: Verifying setup...${NC}"

echo "Trigger details:"
gcloud eventarc triggers describe ${TRIGGER_NAME} \
    --location=${REGION} \
    --project=${PROJECT_ID} \
    --format="table(displayName, destination.workflow, eventFilters)"

echo ""
echo -e "${GREEN}=== Setup Complete ===${NC}"
echo ""
echo "USD Assembly Pipeline Configuration:"
echo "  Trigger: ${TRIGGER_NAME}"
echo "  Workflow: ${WORKFLOW_NAME}"
echo "  Bucket: ${BUCKET}"
echo "  Service Account: ${SA_EMAIL}"
echo ""
echo "The pipeline will trigger when .stage1_complete markers are uploaded to:"
echo "  gs://${BUCKET}/scenes/{scene_id}/assets/.stage1_complete"
echo ""
echo "This will run: GLB->USDZ conversion, Simready preparation, USD assembly,"
echo "  Replicator bundle generation, and Isaac Lab task generation."
echo ""
echo "To test manually:"
echo "  echo '{}' | gsutil cp - gs://${BUCKET}/scenes/test_scene/assets/.stage1_complete"
echo ""
echo "To view trigger:"
echo "  gcloud eventarc triggers describe ${TRIGGER_NAME} --location=${REGION}"
echo ""
echo "To view workflow executions:"
echo "  gcloud workflows executions list ${WORKFLOW_NAME} --location=${REGION}"
echo ""
