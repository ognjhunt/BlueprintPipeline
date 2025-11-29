#!/bin/bash
set -e

# Deploy assets-plan-job to Google Cloud Run
# This job creates scene_assets.json which triggers all downstream pipelines

PROJECT_ID="${PROJECT_ID:-blueprint-8c1ca}"
REGION="${REGION:-us-central1}"
JOB_NAME="assets-plan-job"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${JOB_NAME}"

echo "=========================================="
echo "Deploying ${JOB_NAME}"
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "=========================================="

# Navigate to the job directory
cd "$(dirname "$0")/assets-plan"

echo ""
echo "Step 1: Building Docker image..."
docker build -t "${IMAGE_NAME}:latest" .

echo ""
echo "Step 2: Pushing image to Google Container Registry..."
docker push "${IMAGE_NAME}:latest"

echo ""
echo "Step 3: Deploying Cloud Run job..."
gcloud run jobs create "${JOB_NAME}" \
  --image="${IMAGE_NAME}:latest" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --max-retries=0 \
  --task-timeout=1800 \
  --memory=2Gi \
  --cpu=2 \
  --set-env-vars="BUCKET=PLACEHOLDER" \
  2>/dev/null || \
gcloud run jobs update "${JOB_NAME}" \
  --image="${IMAGE_NAME}:latest" \
  --region="${REGION}" \
  --project="${PROJECT_ID}"

echo ""
echo "=========================================="
echo "✅ ${JOB_NAME} deployed successfully!"
echo "=========================================="
echo ""
echo "Verify deployment:"
echo "  gcloud run jobs describe ${JOB_NAME} --region=${REGION}"
echo ""
echo "Test execution:"
echo "  gcloud run jobs execute ${JOB_NAME} --region=${REGION} \\"
echo "    --update-env-vars=BUCKET=your-bucket,SCENE_ID=test_scene,LAYOUT_PREFIX=scenes/test_scene/layout,MULTIVIEW_PREFIX=scenes/test_scene/multiview,ASSETS_PREFIX=scenes/test_scene/assets"
echo ""
