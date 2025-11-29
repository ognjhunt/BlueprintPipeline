#!/bin/bash
set -e

# Deploy all Cloud Run jobs for the Blueprint Pipeline
# Run this script to deploy or update all jobs

PROJECT_ID="${PROJECT_ID:-blueprint-8c1ca}"
REGION="${REGION:-us-central1}"

echo "=========================================="
echo "Blueprint Pipeline - Job Deployment"
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "=========================================="

# Function to deploy a single job
deploy_job() {
  local JOB_DIR=$1
  local JOB_NAME=$2
  local MEMORY=${3:-2Gi}
  local CPU=${4:-2}
  local TIMEOUT=${5:-1800}

  if [ ! -d "${JOB_DIR}" ]; then
    echo "⚠️  Skipping ${JOB_NAME} (directory not found)"
    return
  fi

  echo ""
  echo "----------------------------------------"
  echo "Deploying: ${JOB_NAME}"
  echo "----------------------------------------"

  IMAGE_NAME="gcr.io/${PROJECT_ID}/${JOB_NAME}"

  cd "${JOB_DIR}"

  echo "Building image..."
  docker build -t "${IMAGE_NAME}:latest" -q .

  echo "Pushing image..."
  docker push "${IMAGE_NAME}:latest" > /dev/null

  echo "Deploying job..."
  gcloud run jobs create "${JOB_NAME}" \
    --image="${IMAGE_NAME}:latest" \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --max-retries=0 \
    --task-timeout="${TIMEOUT}" \
    --memory="${MEMORY}" \
    --cpu="${CPU}" \
    --quiet \
    2>/dev/null || \
  gcloud run jobs update "${JOB_NAME}" \
    --image="${IMAGE_NAME}:latest" \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --quiet

  echo "✅ ${JOB_NAME} deployed"

  cd - > /dev/null
}

# Change to repo root
cd "$(dirname "$0")"

# Deploy critical pipeline jobs (in order of pipeline flow)
echo ""
echo "=== Core Pipeline Jobs ==="

deploy_job "scene-da3-job" "scene-da3-job" "4Gi" "4" "3600"
deploy_job "layout-job" "layout-job" "2Gi" "2" "1800"
deploy_job "scale-job" "scale-job" "2Gi" "1" "600"
deploy_job "objects-job" "objects-job" "2Gi" "2" "1800"
deploy_job "multiview-job" "multiview-job" "4Gi" "4" "3600"

# CRITICAL: This was missing!
echo ""
echo "=== CRITICAL: assets-plan-job ==="
deploy_job "assets-plan" "assets-plan-job" "2Gi" "2" "1800"

echo ""
echo "=== Asset Processing Jobs ==="

deploy_job "hunyuan-job" "hunyuan-job" "8Gi" "4" "7200"
deploy_job "sam3d-job" "sam3d-job" "8Gi" "4" "7200"
deploy_job "interactive-job" "interactive-job" "4Gi" "2" "3600"

echo ""
echo "=== Final Assembly Jobs ==="

deploy_job "simready-job" "simready-job" "2Gi" "2" "1800"
deploy_job "usd-assembly-job" "usd-assembly-job" "4Gi" "2" "3600"

echo ""
echo "=== Optional Jobs ==="

deploy_job "seg-job" "seg-job" "4Gi" "4" "3600"
deploy_job "meshy-job" "meshy-job" "4Gi" "2" "3600"

echo ""
echo "=========================================="
echo "✅ All jobs deployed successfully!"
echo "=========================================="
echo ""
echo "Verify deployments:"
echo "  gcloud run jobs list --region=${REGION}"
echo ""
echo "Next steps:"
echo "  1. Deploy Eventarc triggers: cd terraform && terraform apply"
echo "  2. Test the pipeline by uploading scene_layout_scaled.json"
echo ""
