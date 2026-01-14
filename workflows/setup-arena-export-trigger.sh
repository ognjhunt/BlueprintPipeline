#!/bin/bash
# =============================================================================
# Setup EventArc triggers for Arena Export Pipeline
# =============================================================================
#
# This script creates THREE EventArc triggers that invoke the arena-export-pipeline
# workflow whenever completion markers are uploaded to GCS:
#
#   1. .usd_complete - USD baseline scene
#   2. .geniesim_complete - Genie Sim export complete (.geniesim_submitted ignored)
#   3. .isaac_lab_complete - Isaac Lab task generation complete
#
# The pipeline exports scenes to Isaac Lab-Arena format for policy evaluation.
#
# Usage:
#   ./setup-arena-export-trigger.sh <project_id> <bucket_name>
#
# Prerequisites:
#   - gcloud CLI authenticated
#   - arena-export-pipeline workflow deployed
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
WORKFLOW_NAME="arena-export-pipeline"
WORKFLOW_SA="workflow-invoker"

echo -e "${GREEN}=== Arena Export EventArc Trigger Setup ===${NC}"
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

    # Wait for propagation
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
# Step 3: Verify workflow exists
# =============================================================================
echo -e "${BLUE}Step 3: Verifying workflow exists...${NC}"

if ! gcloud workflows describe ${WORKFLOW_NAME} --location=${REGION} --project=${PROJECT_ID} 2>/dev/null; then
    echo -e "${YELLOW}Workflow ${WORKFLOW_NAME} not found. Deploying...${NC}"
    gcloud workflows deploy ${WORKFLOW_NAME} \
        --location=${REGION} \
        --source=arena-export-pipeline.yaml \
        --description="Arena export pipeline (triggered by .usd_complete, .geniesim_complete, or .isaac_lab_complete; ignores .geniesim_submitted)" \
        --service-account=${SA_EMAIL} \
        --project=${PROJECT_ID}
else
    echo "Workflow ${WORKFLOW_NAME} exists"
fi
echo ""

# =============================================================================
# Step 4: Create EventArc triggers (3 triggers for 3 completion markers)
# =============================================================================
echo -e "${BLUE}Step 4: Creating EventArc triggers...${NC}"

# Helper function to create trigger
create_trigger() {
    local trigger_name=$1
    local description=$2
    local pattern=$3

    if gcloud eventarc triggers describe ${trigger_name} --location=${REGION} --project=${PROJECT_ID} 2>/dev/null; then
        echo -e "${YELLOW}Trigger ${trigger_name} already exists. Deleting...${NC}"
        gcloud eventarc triggers delete ${trigger_name} \
            --location=${REGION} \
            --project=${PROJECT_ID} \
            --quiet
        sleep 3
    fi

    echo "Creating ${trigger_name}..."
    gcloud eventarc triggers create ${trigger_name} \
        --location=${REGION} \
        --project=${PROJECT_ID} \
        --service-account="${SA_EMAIL}" \
        --destination-workflow=${WORKFLOW_NAME} \
        --destination-workflow-location=${REGION} \
        --event-filters="type=google.cloud.storage.object.v1.finalized" \
        --event-filters="bucket=${BUCKET}" \
        --event-filters="name=${pattern}" \
        --event-data-content-type="application/json"

    echo -e "${GREEN}  ${trigger_name} created${NC}"
}

# Create triggers for each completion marker
create_trigger "arena-export-usd-trigger" \
    "Arena export on USD completion" \
    "^scenes/.+/usd/\\.usd_complete$"

create_trigger "arena-export-geniesim-trigger" \
    "Arena export on Genie Sim completion" \
    "^scenes/.+/geniesim/\\.geniesim_complete$"

create_trigger "arena-export-isaac-lab-trigger" \
    "Arena export on Isaac Lab completion" \
    "^scenes/.+/isaac_lab/\\.isaac_lab_complete$"

echo ""

# =============================================================================
# Step 5: Verify setup
# =============================================================================
echo -e "${BLUE}Step 5: Verifying setup...${NC}"

echo "Trigger details:"
gcloud eventarc triggers list \
    --location=${REGION} \
    --project=${PROJECT_ID} \
    --filter="displayName:*arena*" \
    --format="table(displayName, destination.workflow, eventFilters)"

echo ""
echo -e "${GREEN}=== Setup Complete ===${NC}"
echo ""
echo "Arena Export Pipeline Configuration:"
echo "  Workflow: ${WORKFLOW_NAME}"
echo "  Bucket: ${BUCKET}"
echo "  Service Account: ${SA_EMAIL}"
echo ""
echo "The pipeline will trigger when ANY of these markers are uploaded:"
echo "  gs://${BUCKET}/scenes/{scene_id}/usd/.usd_complete"
echo "  gs://${BUCKET}/scenes/{scene_id}/geniesim/.geniesim_complete"
echo "  gs://${BUCKET}/scenes/{scene_id}/isaac_lab/.isaac_lab_complete"
echo "  (Note: .geniesim_submitted does not trigger Arena export)"
echo ""
echo "This will generate Arena format for policy evaluation infrastructure."
echo ""
echo "To test manually:"
echo "  echo '{}' | gsutil cp - gs://${BUCKET}/scenes/test_scene/usd/.usd_complete"
echo ""
echo "To view triggers:"
echo "  gcloud eventarc triggers list --location=${REGION} --filter='displayName:*arena*'"
echo ""
echo "To view workflow executions:"
echo "  gcloud workflows executions list ${WORKFLOW_NAME} --location=${REGION}"
echo ""
