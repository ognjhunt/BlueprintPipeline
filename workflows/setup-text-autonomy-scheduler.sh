#!/bin/bash
# =============================================================================
# Setup Text Autonomy Daily Workflow + Cloud Scheduler
# =============================================================================
#
# Usage:
#   ./setup-text-autonomy-scheduler.sh <project_id> [bucket] [region]
#
# Defaults:
#   Schedule: 09:00 America/New_York (daily)
#   Daily quota: 1
# =============================================================================

set -e

PROJECT_ID=${1:-$(gcloud config get-value project)}
BUCKET=${2:-"${PROJECT_ID}-blueprint-scenes"}
REGION=${3:-"us-central1"}

WORKFLOW_NAME="text-autonomy-daily"
WORKFLOW_SA=${WORKFLOW_SA:-"workflow-invoker"}
WORKFLOW_SA_EMAIL="${WORKFLOW_SA}@${PROJECT_ID}.iam.gserviceaccount.com"
SCHEDULER_JOB=${TEXT_AUTONOMY_SCHEDULER_JOB:-"text-autonomy-daily"}
SCHEDULER_SCHEDULE=${TEXT_AUTONOMY_SCHEDULER_CRON:-"0 9 * * *"}
SCHEDULER_TIMEZONE=${TEXT_AUTONOMY_TIMEZONE:-"America/New_York"}
SCHEDULER_PAUSED=${TEXT_AUTONOMY_SCHEDULER_PAUSED:-"false"}

TEXT_AUTONOMY_STATE_PREFIX=${TEXT_AUTONOMY_STATE_PREFIX:-"automation/text_daily"}
TEXT_DAILY_QUOTA=${TEXT_DAILY_QUOTA:-"1"}
TEXT_DAILY_PAUSE_AFTER_CONSEC_FAILS=${TEXT_DAILY_PAUSE_AFTER_CONSEC_FAILS:-"3"}
TEXT_AUTONOMY_EMITTER_JOB_NAME=${TEXT_AUTONOMY_EMITTER_JOB_NAME:-"text-request-emitter-job"}
TEXT_AUTONOMY_EMITTER_TIMEOUT_SECONDS=${TEXT_AUTONOMY_EMITTER_TIMEOUT_SECONDS:-"900"}
TEXT_AUTONOMY_PROVIDER_POLICY=${TEXT_AUTONOMY_PROVIDER_POLICY:-"openrouter_qwen_primary"}
TEXT_AUTONOMY_TEXT_BACKEND=${TEXT_AUTONOMY_TEXT_BACKEND:-"hybrid_serial"}
TEXT_AUTONOMY_QUALITY_TIER=${TEXT_AUTONOMY_QUALITY_TIER:-"premium"}
TEXT_AUTONOMY_SEED_COUNT=${TEXT_AUTONOMY_SEED_COUNT:-"1"}
TEXT_AUTONOMY_SOURCE_WAIT_TIMEOUT_SECONDS=${TEXT_AUTONOMY_SOURCE_WAIT_TIMEOUT_SECONDS:-"21600"}
TEXT_AUTONOMY_SOURCE_WAIT_POLL_SECONDS=${TEXT_AUTONOMY_SOURCE_WAIT_POLL_SECONDS:-"30"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

printf "=== Text Autonomy Daily Setup ===\n"
printf "Project: %s\n" "${PROJECT_ID}"
printf "Bucket: %s\n" "${BUCKET}"
printf "Region: %s\n" "${REGION}"
printf "Schedule: %s (%s)\n" "${SCHEDULER_SCHEDULE}" "${SCHEDULER_TIMEZONE}"
printf "Paused: %s\n\n" "${SCHEDULER_PAUSED}"

printf "Step 1: Enable APIs\n"
gcloud services enable \
  workflows.googleapis.com \
  workflowexecutions.googleapis.com \
  cloudscheduler.googleapis.com \
  run.googleapis.com \
  storage.googleapis.com \
  --project="${PROJECT_ID}"

printf "Step 2: Ensure workflow service account exists\n"
if ! gcloud iam service-accounts describe "${WORKFLOW_SA_EMAIL}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud iam service-accounts create "${WORKFLOW_SA}" \
    --display-name="Workflow Invoker SA" \
    --project="${PROJECT_ID}"
  sleep 5
fi

for ROLE in \
  "roles/workflows.invoker" \
  "roles/run.invoker" \
  "roles/storage.objectAdmin" \
  "roles/logging.logWriter" \
  "roles/eventarc.eventReceiver"; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${WORKFLOW_SA_EMAIL}" \
    --role="${ROLE}" \
    --quiet >/dev/null

done

printf "Step 3: Deploy workflow\n"
gcloud workflows deploy "${WORKFLOW_NAME}" \
  --location="${REGION}" \
  --source="${SCRIPT_DIR}/text-autonomy-daily.yaml" \
  --service-account="${WORKFLOW_SA_EMAIL}" \
  --set-env-vars="PRIMARY_BUCKET=${BUCKET},TEXT_AUTONOMY_STATE_PREFIX=${TEXT_AUTONOMY_STATE_PREFIX},TEXT_AUTONOMY_TIMEZONE=${SCHEDULER_TIMEZONE},TEXT_DAILY_QUOTA=${TEXT_DAILY_QUOTA},TEXT_DAILY_PAUSE_AFTER_CONSEC_FAILS=${TEXT_DAILY_PAUSE_AFTER_CONSEC_FAILS},TEXT_AUTONOMY_EMITTER_JOB_NAME=${TEXT_AUTONOMY_EMITTER_JOB_NAME},TEXT_AUTONOMY_EMITTER_TIMEOUT_SECONDS=${TEXT_AUTONOMY_EMITTER_TIMEOUT_SECONDS},TEXT_AUTONOMY_PROVIDER_POLICY=${TEXT_AUTONOMY_PROVIDER_POLICY},TEXT_AUTONOMY_TEXT_BACKEND=${TEXT_AUTONOMY_TEXT_BACKEND},TEXT_AUTONOMY_QUALITY_TIER=${TEXT_AUTONOMY_QUALITY_TIER},TEXT_AUTONOMY_SEED_COUNT=${TEXT_AUTONOMY_SEED_COUNT},TEXT_AUTONOMY_SOURCE_WAIT_TIMEOUT_SECONDS=${TEXT_AUTONOMY_SOURCE_WAIT_TIMEOUT_SECONDS},TEXT_AUTONOMY_SOURCE_WAIT_POLL_SECONDS=${TEXT_AUTONOMY_SOURCE_WAIT_POLL_SECONDS}" \
  --project="${PROJECT_ID}"

printf "Step 4: Configure scheduler\n"
SCHEDULER_URI="https://workflowexecutions.googleapis.com/v1/projects/${PROJECT_ID}/locations/${REGION}/workflows/${WORKFLOW_NAME}/executions"

if gcloud scheduler jobs describe "${SCHEDULER_JOB}" --location="${REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud scheduler jobs delete "${SCHEDULER_JOB}" --location="${REGION}" --project="${PROJECT_ID}" --quiet
fi

gcloud scheduler jobs create http "${SCHEDULER_JOB}" \
  --location="${REGION}" \
  --schedule="${SCHEDULER_SCHEDULE}" \
  --time-zone="${SCHEDULER_TIMEZONE}" \
  --uri="${SCHEDULER_URI}" \
  --http-method=POST \
  --oauth-service-account-email="${WORKFLOW_SA_EMAIL}" \
  --headers="Content-Type=application/json" \
  --message-body="{\"argument\": \"{\\\"bucket\\\": \\\"${BUCKET}\\\"}\"}" \
  --project="${PROJECT_ID}"

if [ "${SCHEDULER_PAUSED}" = "true" ]; then
  gcloud scheduler jobs pause "${SCHEDULER_JOB}" --location="${REGION}" --project="${PROJECT_ID}"
else
  gcloud scheduler jobs resume "${SCHEDULER_JOB}" --location="${REGION}" --project="${PROJECT_ID}" || true
fi

printf "\nSetup complete.\n"
printf "Workflow: %s\n" "${WORKFLOW_NAME}"
printf "Scheduler: %s\n" "${SCHEDULER_JOB}"
printf "Cron: %s\n" "${SCHEDULER_SCHEDULE}"
printf "Timezone: %s\n" "${SCHEDULER_TIMEZONE}"
printf "Quota/day: %s\n" "${TEXT_DAILY_QUOTA}"
