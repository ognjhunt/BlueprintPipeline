#!/bin/bash
# =============================================================================
# Setup Webhook endpoint for Genie Sim Import Pipeline
# =============================================================================
#
# This script sets up a Cloud Run service that accepts webhooks from Genie Sim 3.0
# when data generation jobs complete, then triggers the genie-sim-import-pipeline.
#
# Genie Sim will send a webhook POST request to:
#   https://geniesim-webhook.${PROJECT_ID}.run.app/webhooks/geniesim/job-complete
#
# With body: {"job_id": "...", "scene_id": "...", "status": "completed"}
#
# The Cloud Run service will invoke the genie-sim-import-pipeline workflow
# to import the generated episodes back into BlueprintPipeline.
#
# Usage:
#   ./setup-genie-sim-import-trigger.sh <project_id>
#
# Prerequisites:
#   - gcloud CLI authenticated
#   - genie-sim-import-pipeline workflow deployed
#   - Cloud Run and Workflows APIs enabled
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
REGION=${2:-"us-central1"}

# Derived values
WORKFLOW_NAME="genie-sim-import-pipeline"
WEBHOOK_SERVICE="geniesim-webhook"
WORKFLOW_SA="workflow-invoker"

echo -e "${GREEN}=== Genie Sim Import Webhook Setup ===${NC}"
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo ""

# =============================================================================
# Step 1: Enable required APIs
# =============================================================================
echo -e "${BLUE}Step 1: Enabling required APIs...${NC}"

gcloud services enable \
    run.googleapis.com \
    workflows.googleapis.com \
    cloudfunctions.googleapis.com \
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
    --role="roles/storage.objectViewer" \
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
        --source=genie-sim-import-pipeline.yaml \
        --description="Genie Sim import pipeline (triggered by Genie Sim webhooks)" \
        --service-account=${SA_EMAIL} \
        --project=${PROJECT_ID}
else
    echo "Workflow ${WORKFLOW_NAME} exists"
fi
echo ""

# =============================================================================
# Step 4: Create Cloud Run webhook service
# =============================================================================
echo -e "${BLUE}Step 4: Creating Cloud Run webhook service...${NC}"

# Create a minimal Cloud Run service that triggers the workflow
# This service will receive webhooks from Genie Sim
echo -e "${YELLOW}Deploying Cloud Run webhook receiver...${NC}"

# For now, we'll provide instructions for manual setup
# since the webhook service needs to be implemented as a Cloud Function or custom service
echo ""
echo -e "${YELLOW}Manual Webhook Setup Required:${NC}"
echo ""
echo "You need to deploy a webhook receiver service. Here are the options:"
echo ""
echo "OPTION 1: Use Google Cloud Functions (Recommended)"
echo "  1. Create a Cloud Function in Python:"
echo ""
cat << 'EOF'
    def geniesim_webhook(request):
        import json
        import functions_framework
        from google.cloud import workflows_v1beta

        request_json = request.get_json()
        project_id = "YOUR_PROJECT_ID"
        region = "us-central1"

        # Invoke workflow
        client = workflows_v1beta.ExecutionsClient()
        parent = f"projects/{project_id}/locations/{region}/workflows/genie-sim-import-pipeline"

        execution = {
            "argument": json.dumps(request_json)
        }

        response = client.create_execution(request={"parent": parent, "execution": execution})
        return {"status": "triggered", "execution": response.name}
EOF
echo ""
echo "  2. Deploy with:"
echo "    gcloud functions deploy geniesim-webhook \\"
echo "      --runtime python39 \\"
echo "      --trigger-http \\"
echo "      --allow-unauthenticated \\"
echo "      --region=${REGION} \\"
echo "      --project=${PROJECT_ID}"
echo ""
echo "OPTION 2: Create a Cloud Run service"
echo "  1. Build a Docker image that handles webhooks"
echo "  2. Deploy to Cloud Run with:"
echo "    gcloud run deploy geniesim-webhook \\"
echo "      --image gcr.io/${PROJECT_ID}/geniesim-webhook \\"
echo "      --region=${REGION} \\"
echo "      --allow-unauthenticated \\"
echo "      --project=${PROJECT_ID}"
echo ""
echo "OPTION 3: Use Pub/Sub for decoupled triggering"
echo "  1. Create a Pub/Sub topic:"
echo "    gcloud pubsub topics create geniesim-job-events --project=${PROJECT_ID}"
echo ""
echo "  2. Create a Cloud Scheduler job to trigger the workflow:"
echo "    gcloud scheduler jobs create pubsub geniesim-import-poller \\"
echo "      --schedule='*/5 * * * *' \\"
echo "      --topic=geniesim-job-events \\"
echo "      --message-body='{}' \\"
echo "      --location=${REGION} \\"
echo "      --project=${PROJECT_ID}"
echo ""

echo ""
echo -e "${GREEN}=== Setup Complete ===${NC}"
echo ""
echo "Genie Sim Import Pipeline Configuration:"
echo "  Workflow: ${WORKFLOW_NAME}"
echo "  Region: ${REGION}"
echo "  Service Account: ${SA_EMAIL}"
echo ""
echo "Next Steps:"
echo "  1. Choose a webhook delivery mechanism (Cloud Function, Cloud Run, or Pub/Sub)"
echo "  2. Configure Genie Sim to send completion webhooks"
echo "  3. Webhook payload should include:"
echo "     - job_id (required)"
echo "     - scene_id (optional)"
echo "     - status (required, should be 'completed')"
echo ""
echo "Example webhook payload:"
echo '  {"job_id": "job-12345", "scene_id": "kitchen_001", "status": "completed"}'
echo ""
echo "To invoke workflow manually:"
echo "  gcloud workflows run ${WORKFLOW_NAME} --location=${REGION} \\"
echo "    --data='{\"job_id\": \"test-job\", \"status\": \"completed\"}'"
echo ""
echo "To view workflow executions:"
echo "  gcloud workflows executions list ${WORKFLOW_NAME} --location=${REGION}"
echo ""
