#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID=${1:-$(gcloud config get-value project)}
REGION=${2:-us-central1}
ARTIFACT_REPO=${3:-blueprint}

if [[ -z "${PROJECT_ID}" ]]; then
  echo "PROJECT_ID is required" >&2
  exit 1
fi

SCENESMITH_SERVICE_NAME=${SCENESMITH_SERVICE_NAME:-scenesmith-service}
SAGE_SERVICE_NAME=${SAGE_SERVICE_NAME:-sage-service}

SCENESMITH_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPO}/${SCENESMITH_SERVICE_NAME}:latest"
SAGE_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPO}/${SAGE_SERVICE_NAME}:latest"

SCENESMITH_MODE=${SCENESMITH_SERVICE_MODE:-internal}
SAGE_MODE=${SAGE_SERVICE_MODE:-internal}
SCENESMITH_TIMEOUT=${SCENESMITH_SERVICE_TIMEOUT_SECONDS:-3600}
SAGE_TIMEOUT=${SAGE_SERVICE_TIMEOUT_SECONDS:-1800}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "Building ${SCENESMITH_SERVICE_NAME} image: ${SCENESMITH_IMAGE}"
gcloud builds submit --project="${PROJECT_ID}" --tag "${SCENESMITH_IMAGE}" -f scenesmith-service/Dockerfile .

echo "Building ${SAGE_SERVICE_NAME} image: ${SAGE_IMAGE}"
gcloud builds submit --project="${PROJECT_ID}" --tag "${SAGE_IMAGE}" -f sage-service/Dockerfile .

echo "Deploying ${SCENESMITH_SERVICE_NAME}"
gcloud run deploy "${SCENESMITH_SERVICE_NAME}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --image="${SCENESMITH_IMAGE}" \
  --allow-unauthenticated \
  --set-env-vars="SCENESMITH_SERVICE_MODE=${SCENESMITH_MODE},SCENESMITH_SERVICE_TIMEOUT_SECONDS=${SCENESMITH_TIMEOUT}"

echo "Deploying ${SAGE_SERVICE_NAME}"
gcloud run deploy "${SAGE_SERVICE_NAME}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --image="${SAGE_IMAGE}" \
  --allow-unauthenticated \
  --set-env-vars="SAGE_SERVICE_MODE=${SAGE_MODE},SAGE_SERVICE_TIMEOUT_SECONDS=${SAGE_TIMEOUT}"

echo "Done."
echo "Set these in source-orchestrator env:"
echo "  SCENESMITH_SERVER_URL=https://$(gcloud run services describe "${SCENESMITH_SERVICE_NAME}" --region="${REGION}" --project="${PROJECT_ID}" --format='value(status.url)')/v1/generate"
echo "  SAGE_SERVER_URL=https://$(gcloud run services describe "${SAGE_SERVICE_NAME}" --region="${REGION}" --project="${PROJECT_ID}" --format='value(status.url)')/v1/refine"
