#!/bin/bash
# =============================================================================
# Setup single EventArc trigger for the Image-to-Scene Orchestrator
# =============================================================================
#
# This script creates ONE EventArc trigger that invokes the
# image-to-scene-orchestrator workflow on any GCS object finalization.
# The orchestrator filters for image uploads internally and drives the
# entire pipeline end-to-end (reconstruction → USD → variation → GenieSim → arena).
#
# Replaces the old multi-trigger chain (setup-image-trigger.sh + setup-usd-assembly-trigger.sh
# + setup-genie-sim-export-trigger.sh + setup-arena-export-trigger.sh).
#
# Usage:
#   ./setup-orchestrator-trigger.sh <project_id> <bucket_name> [workflow_region]
#
# Prerequisites:
#   - gcloud CLI authenticated
#   - Cloud Build, Workflows, EventArc, Compute Engine APIs enabled
#
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

# Eventarc trigger must be in the same location as the GCS bucket.
# blueprint-8c1ca.appspot.com is multi-region US, so trigger goes in "us".
TRIGGER_LOCATION="us"

TRIGGER_NAME="image-upload-orchestrator-trigger"
WORKFLOW_NAME="image-to-scene-orchestrator"
WORKFLOW_SA="workflow-invoker"

echo -e "${GREEN}=== Image-to-Scene Orchestrator Trigger Setup ===${NC}"
echo "Project:           ${PROJECT_ID}"
echo "Bucket:            ${BUCKET}"
echo "Trigger location:  ${TRIGGER_LOCATION}"
echo "Workflow region:   ${WORKFLOW_REGION}"
echo ""

# =============================================================================
# Step 1: Enable required APIs
# =============================================================================
echo -e "${BLUE}Step 1: Enabling required APIs...${NC}"

gcloud services enable \
    eventarc.googleapis.com \
    workflows.googleapis.com \
    workflowexecutions.googleapis.com \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    storage.googleapis.com \
    compute.googleapis.com \
    --project=${PROJECT_ID}

echo -e "${GREEN}APIs enabled${NC}"
echo ""

# =============================================================================
# Step 2: Create/verify service account
# =============================================================================
echo -e "${BLUE}Step 2: Setting up service account...${NC}"

SA_EMAIL="${WORKFLOW_SA}@${PROJECT_ID}.iam.gserviceaccount.com"
PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
CLOUD_BUILD_SA="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"

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
    "roles/compute.instanceAdmin.v1" \
    "roles/cloudbuild.builds.editor" \
    "roles/eventarc.eventReceiver"; do
    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
        --member="serviceAccount:${SA_EMAIL}" \
        --role="${ROLE}" \
        --quiet 2>&1 | tail -1
done

# Cloud Build SA needs compute SSH access
for ROLE in \
    "roles/compute.instanceAdmin.v1" \
    "roles/compute.osAdminLogin" \
    "roles/logging.logWriter"; do
    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
        --member="serviceAccount:${CLOUD_BUILD_SA}" \
        --role="${ROLE}" \
        --quiet 2>&1 | tail -1
done

# GCS service agent needs pubsub publisher for Eventarc
GCS_SA="service-${PROJECT_NUMBER}@gs-project-accounts.iam.gserviceaccount.com"
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${GCS_SA}" \
    --role="roles/pubsub.publisher" \
    --quiet 2>&1 | tail -1

echo -e "${GREEN}Permissions granted${NC}"
echo ""

# =============================================================================
# Step 3: Deploy orchestrator workflow
# =============================================================================
echo -e "${BLUE}Step 3: Deploying orchestrator workflow...${NC}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

gcloud workflows deploy ${WORKFLOW_NAME} \
    --location=${WORKFLOW_REGION} \
    --source="${SCRIPT_DIR}/image-to-scene-orchestrator.yaml" \
    --service-account=${SA_EMAIL} \
    --project=${PROJECT_ID}

echo -e "${GREEN}Workflow deployed${NC}"
echo ""

# =============================================================================
# Step 4: Delete existing trigger if present
# =============================================================================
echo -e "${BLUE}Step 4: Checking for existing trigger...${NC}"

if gcloud eventarc triggers describe ${TRIGGER_NAME} --location=${TRIGGER_LOCATION} --project=${PROJECT_ID} 2>/dev/null; then
    echo -e "${YELLOW}Trigger ${TRIGGER_NAME} already exists. Deleting and recreating...${NC}"
    gcloud eventarc triggers delete ${TRIGGER_NAME} \
        --location=${TRIGGER_LOCATION} \
        --project=${PROJECT_ID} \
        --quiet
    sleep 5
fi

# =============================================================================
# Step 5: Create EventArc trigger
# =============================================================================
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

echo -e "${GREEN}EventArc trigger created${NC}"
echo ""

# =============================================================================
# Step 6: Verify setup
# =============================================================================
echo -e "${BLUE}Step 6: Verifying setup...${NC}"

echo "Trigger details:"
gcloud eventarc triggers describe ${TRIGGER_NAME} \
    --location=${TRIGGER_LOCATION} \
    --project=${PROJECT_ID} \
    --format="table(displayName, destination.workflow, eventFilters)"

echo ""
echo -e "${GREEN}=== Setup Complete ===${NC}"
echo ""
echo "Image-to-Scene Orchestrator Configuration:"
echo "  Trigger:          ${TRIGGER_NAME} (location: ${TRIGGER_LOCATION})"
echo "  Workflow:         ${WORKFLOW_NAME} (location: ${WORKFLOW_REGION})"
echo "  Bucket:           ${BUCKET}"
echo "  Service Account:  ${SA_EMAIL}"
echo ""
echo "The orchestrator triggers on ANY file upload to the bucket."
echo "It filters internally for: scenes/{scene_id}/images/{name}.{png|jpg|jpeg}"
echo "Non-matching files are skipped instantly."
echo ""
echo "Pipeline stages (all automatic):"
echo "  1. VM Reconstruction (run_pipeline_gcs.sh on isaac-sim-ubuntu)"
echo "  2. USD Assembly (usd-assembly-pipeline)"
echo "  3. Variation Assets (variation-assets-pipeline)"
echo "  4. GenieSim Export (genie-sim-export-pipeline)"
echo "  5. Arena Export (arena-export-pipeline, non-blocking)"
echo ""
echo "To test:"
echo "  gsutil cp photo.jpeg gs://${BUCKET}/scenes/test_scene/images/kitchen.jpeg"
echo ""
echo "To monitor:"
echo "  gcloud workflows executions list ${WORKFLOW_NAME} --location=${WORKFLOW_REGION} --limit=5"
echo ""
