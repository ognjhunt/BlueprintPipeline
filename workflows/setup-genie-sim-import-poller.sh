#!/bin/bash
# =============================================================================
# Setup Genie Sim Import Poller (Fallback)
# =============================================================================
#
# Deploys the genie-sim-import-poller workflow and schedules it via
# Cloud Scheduler to trigger imports when webhooks are unavailable.
#
# Usage:
#   ./setup-genie-sim-import-poller.sh <project_id> [bucket] [region]
#
# =============================================================================

set -e

PROJECT_ID=${1:-$(gcloud config get-value project)}
BUCKET=${2:-"${PROJECT_ID}-blueprint-scenes"}
REGION=${3:-"us-central1"}

WORKFLOW_NAME="genie-sim-import-poller"
WORKFLOW_SA="workflow-invoker"
SA_EMAIL="${WORKFLOW_SA}@${PROJECT_ID}.iam.gserviceaccount.com"

print_header() {
    local title=$1
    echo ""
    echo "=== ${title} ==="
}

print_header "Deploying workflow"

gcloud services enable \
    workflows.googleapis.com \
    cloudscheduler.googleapis.com \
    --project=${PROJECT_ID}

if ! gcloud workflows describe ${WORKFLOW_NAME} --location=${REGION} --project=${PROJECT_ID} 2>/dev/null; then
    gcloud workflows deploy ${WORKFLOW_NAME} \
        --location=${REGION} \
        --source=genie-sim-import-poller.yaml \
        --description="Genie Sim import poller (fallback)" \
        --service-account=${SA_EMAIL} \
        --project=${PROJECT_ID}
fi

print_header "Creating Cloud Scheduler job"

SCHEDULER_JOB="genie-sim-import-poller"
SCHEDULER_URI="https://workflowexecutions.googleapis.com/v1/projects/${PROJECT_ID}/locations/${REGION}/workflows/${WORKFLOW_NAME}/executions"

if gcloud scheduler jobs describe ${SCHEDULER_JOB} --location=${REGION} --project=${PROJECT_ID} >/dev/null 2>&1; then
    gcloud scheduler jobs delete ${SCHEDULER_JOB} --location=${REGION} --project=${PROJECT_ID} --quiet
fi

gcloud scheduler jobs create http ${SCHEDULER_JOB} \
    --location=${REGION} \
    --schedule="*/10 * * * *" \
    --uri="${SCHEDULER_URI}" \
    --http-method=POST \
    --oauth-service-account-email=${SA_EMAIL} \
    --headers="Content-Type=application/json" \
    --message-body="{\"argument\": \"{\\\"bucket\\\": \\\"${BUCKET}\\\"}\"}" \
    --project=${PROJECT_ID}

print_header "Done"

echo "Poller workflow: ${WORKFLOW_NAME}"
echo "Schedule: every 10 minutes"
echo "Bucket: ${BUCKET}"
