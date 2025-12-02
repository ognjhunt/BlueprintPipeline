#!/usr/bin/env bash
set -euo pipefail

# Update Cloud Run Job timeout to maximum allowed (1 hour for GPU jobs)
# Note: GPU-attached jobs have a hard limit of 1 hour (3600 seconds)
# We rely on performance optimizations to complete within this window

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-blueprint-8c1ca}"
REGION="${REGION:-us-central1}"
JOB_NAME="${JOB_NAME:-hunyuan-job}"

echo "⚠️  Note: GPU-attached Cloud Run jobs have a maximum timeout of 1 hour"
echo "Configuring job for maximum allowed timeout..."
echo ""
echo "Updating: ${JOB_NAME}"
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"

gcloud run jobs update "${JOB_NAME}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --task-timeout=3600s \
  --max-retries=0

echo ""
echo "✓ Job timeout set to 3600 seconds (1 hour - maximum for GPU jobs)"
echo ""
echo "Next steps to speed up processing:"
echo "1. Reduce texture quality (30-50% faster):"
echo "   gcloud run jobs update ${JOB_NAME} --region=${REGION} --set-env-vars=HUNYUAN_RENDER_SIZE=512,HUNYUAN_TEXTURE_SIZE=1024"
echo ""
echo "2. Optional - disable USDZ export if not needed:"
echo "   gcloud run jobs update ${JOB_NAME} --region=${REGION} --update-env-vars=ENABLE_USDZ_EXPORT=0"
echo ""
echo "To verify:"
echo "  gcloud run jobs describe ${JOB_NAME} --region=${REGION} --project=${PROJECT_ID}"
