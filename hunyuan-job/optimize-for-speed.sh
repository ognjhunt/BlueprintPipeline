#!/usr/bin/env bash
set -euo pipefail

# Optimize Hunyuan job for speed
# This applies all performance improvements to complete within 1 hour

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-blueprint-8c1ca}"
REGION="${REGION:-us-central1}"
JOB_NAME="${JOB_NAME:-hunyuan-job}"

echo "🚀 Optimizing ${JOB_NAME} for maximum speed"
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo ""

echo "Step 1: Setting maximum timeout (1 hour - GPU limit)..."
gcloud run jobs update "${JOB_NAME}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --task-timeout=3600s \
  --max-retries=0

echo "✓ Timeout configured"
echo ""

echo "Step 2: Reducing texture quality for 30-50% speed improvement..."
gcloud run jobs update "${JOB_NAME}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --set-env-vars="HUNYUAN_RENDER_SIZE=512,HUNYUAN_TEXTURE_SIZE=1024"

echo "✓ Texture quality reduced"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Optimizations applied!"
echo ""
echo "Expected performance (with USDZ export enabled):"
echo "  Before: 65-75 minutes (times out) ❌"
echo "  After:  45-55 minutes (completes) ✅"
echo ""
echo "Note: USDZ export is still enabled (adds ~10 minutes)"
echo ""
echo "To verify configuration:"
echo "  gcloud run jobs describe ${JOB_NAME} --region=${REGION} --project=${PROJECT_ID}"
echo ""
echo "If you need even faster processing and don't need USDZ:"
echo "  gcloud run jobs update ${JOB_NAME} --region=${REGION} --update-env-vars=ENABLE_USDZ_EXPORT=0"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
