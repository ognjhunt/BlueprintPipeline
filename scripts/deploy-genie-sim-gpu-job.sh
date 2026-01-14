#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID=${PROJECT_ID:-$(gcloud config get-value project)}
BUCKET=${BUCKET:-"${PROJECT_ID}-blueprint-scenes"}
SCENE_ID=${SCENE_ID:-""}
REGION=${REGION:-"us-central1"}
ZONE=${ZONE:-"${REGION}-a"}
GKE_CLUSTER=${GKE_CLUSTER:-"blueprint-cluster"}
NAMESPACE=${NAMESPACE:-"blueprint"}
JOB_NAME=${JOB_NAME:-"genie-sim-gpu-${SCENE_ID:-local}-$(date +%s)"}

if [[ -z "${PROJECT_ID}" ]]; then
  echo "PROJECT_ID is required." >&2
  exit 1
fi

if [[ -z "${SCENE_ID}" ]]; then
  echo "SCENE_ID is required (set SCENE_ID=...)." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"

export PROJECT_ID
export BUCKET
export SCENE_ID
export JOB_NAME

printf "Deploying Genie Sim GPU job to GKE...\n"
printf "  Project:    %s\n" "${PROJECT_ID}"
printf "  Cluster:    %s\n" "${GKE_CLUSTER}"
printf "  Zone:       %s\n" "${ZONE}"
printf "  Namespace:  %s\n" "${NAMESPACE}"
printf "  Job name:   %s\n" "${JOB_NAME}"


gcloud container clusters get-credentials "${GKE_CLUSTER}" \
  --zone "${ZONE}" \
  --project "${PROJECT_ID}"

kubectl apply -n "${NAMESPACE}" -f "${REPO_ROOT}/k8s/namespace-setup.yaml"

envsubst < "${REPO_ROOT}/k8s/genie-sim-gpu-job.yaml" | kubectl apply -n "${NAMESPACE}" -f -

printf "Genie Sim GPU job submitted: %s\n" "${JOB_NAME}"
