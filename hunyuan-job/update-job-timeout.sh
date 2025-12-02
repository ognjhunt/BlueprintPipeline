#!/usr/bin/env bash
set -euo pipefail

# Update Cloud Run Job timeout to 2 hours (7200 seconds)
# This fixes the timeout issue where jobs were being killed after 1 hour

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-blueprint-8c1ca}"
REGION="${REGION:-us-central1}"
JOB_NAME="${JOB_NAME:-hunyuan-job}"

echo "Updating timeout for Cloud Run Job: ${JOB_NAME}"
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"

gcloud run jobs update "${JOB_NAME}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --timeout=7200 \
  --max-retries=0

echo "✓ Job timeout updated to 7200 seconds (2 hours)"
echo ""
echo "To verify:"
echo "  gcloud run jobs describe ${JOB_NAME} --region=${REGION} --project=${PROJECT_ID}"
