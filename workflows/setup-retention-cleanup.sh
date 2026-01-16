#!/bin/bash
# =============================================================================
# Setup Retention Cleanup Workflow
# =============================================================================
#
# Deploys the retention-cleanup workflow and schedules it via Cloud Scheduler.
#
# Usage:
#   ./setup-retention-cleanup.sh <project_id> [bucket] [region]
#
# =============================================================================

set -e

PROJECT_ID=${1:-$(gcloud config get-value project)}
BUCKET=${2:-"${PROJECT_ID}-blueprint-scenes"}
REGION=${3:-"us-central1"}

WORKFLOW_NAME="retention-cleanup"
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
        --source=retention-cleanup.yaml \
        --description="Pipeline retention cleanup" \
        --service-account=${SA_EMAIL} \
        --project=${PROJECT_ID}
fi

print_header "Creating Cloud Scheduler job"

SCHEDULER_JOB="retention-cleanup-daily"
SCHEDULER_URI="https://workflowexecutions.googleapis.com/v1/projects/${PROJECT_ID}/locations/${REGION}/workflows/${WORKFLOW_NAME}/executions"

if gcloud scheduler jobs describe ${SCHEDULER_JOB} --location=${REGION} --project=${PROJECT_ID} >/dev/null 2>&1; then
    gcloud scheduler jobs delete ${SCHEDULER_JOB} --location=${REGION} --project=${PROJECT_ID} --quiet
fi

gcloud scheduler jobs create http ${SCHEDULER_JOB} \
    --location=${REGION} \
    --schedule="0 3 * * *" \
    --uri="${SCHEDULER_URI}" \
    --http-method=POST \
    --oauth-service-account-email=${SA_EMAIL} \
    --headers="Content-Type=application/json" \
    --message-body="{\"argument\": \"{\\\"bucket\\\": \\\"${BUCKET}\\\"}\"}" \
    --project=${PROJECT_ID}

print_header "Done"

echo "Workflow: ${WORKFLOW_NAME}"
echo "Schedule: daily at 03:00"
echo "Bucket: ${BUCKET}"
