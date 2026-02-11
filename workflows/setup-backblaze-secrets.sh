#!/bin/bash
# =============================================================================
# Securely store Backblaze B2 credentials in Secret Manager.
# =============================================================================
#
# Usage:
#   ./setup-backblaze-secrets.sh <project_id> [key_id_secret_name] [app_key_secret_name]
#
# Optional env overrides:
#   B2_KEY_ID_VALUE
#   B2_APPLICATION_KEY_VALUE
#

set -euo pipefail

PROJECT_ID=${1:-$(gcloud config get-value project)}
KEY_ID_SECRET=${2:-"b2-key-id"}
APP_KEY_SECRET=${3:-"b2-application-key"}

if [ -z "${PROJECT_ID}" ]; then
  echo "ERROR: project id is required."
  exit 1
fi

if [ -z "${B2_KEY_ID_VALUE:-}" ]; then
  read -r -p "Backblaze B2 Key ID: " B2_KEY_ID_VALUE
fi

if [ -z "${B2_APPLICATION_KEY_VALUE:-}" ]; then
  read -r -s -p "Backblaze B2 Application Key: " B2_APPLICATION_KEY_VALUE
  echo ""
fi

if [ -z "${B2_KEY_ID_VALUE}" ] || [ -z "${B2_APPLICATION_KEY_VALUE}" ]; then
  echo "ERROR: both key ID and application key are required."
  exit 1
fi

gcloud secrets create "${KEY_ID_SECRET}" \
  --project="${PROJECT_ID}" \
  --replication-policy="automatic" >/dev/null 2>&1 || true

gcloud secrets create "${APP_KEY_SECRET}" \
  --project="${PROJECT_ID}" \
  --replication-policy="automatic" >/dev/null 2>&1 || true

printf '%s' "${B2_KEY_ID_VALUE}" | gcloud secrets versions add "${KEY_ID_SECRET}" \
  --project="${PROJECT_ID}" \
  --data-file=- >/dev/null

printf '%s' "${B2_APPLICATION_KEY_VALUE}" | gcloud secrets versions add "${APP_KEY_SECRET}" \
  --project="${PROJECT_ID}" \
  --data-file=- >/dev/null

echo "Stored Backblaze credentials in Secret Manager:"
echo "  key id secret: ${KEY_ID_SECRET}"
echo "  app key secret: ${APP_KEY_SECRET}"
echo ""
echo "Next:"
echo "  B2_KEY_ID_SECRET=${KEY_ID_SECRET} B2_APPLICATION_KEY_SECRET=${APP_KEY_SECRET} \\"
echo "  bash workflows/setup-asset-replication-trigger.sh ${PROJECT_ID} <bucket> <region>"
