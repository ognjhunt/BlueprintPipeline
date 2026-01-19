#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID=${PROJECT_ID:-""}
BUCKET=${BUCKET:-""}
SCENE_ID=${SCENE_ID:-""}
REGION=${REGION:-"us-central1"}
ZONE=${ZONE:-"${REGION}-a"}
GKE_CLUSTER=${GKE_CLUSTER:-"blueprint-cluster"}
NAMESPACE=${NAMESPACE:-"blueprint"}
JOB_NAME=${JOB_NAME:-"genie-sim-local-${SCENE_ID:-local}-$(date +%s)"}

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud is required but was not found. Install the Google Cloud SDK and ensure gcloud is on your PATH." >&2
  exit 1
fi

if ! command -v kubectl >/dev/null 2>&1; then
  echo "kubectl is required but was not found. Install kubectl and ensure it is on your PATH." >&2
  exit 1
fi

if [[ -z "${PROJECT_ID}" ]]; then
  PROJECT_ID=$(gcloud config get-value project 2>/dev/null || true)
fi

if [[ -z "${PROJECT_ID}" ]]; then
  echo "PROJECT_ID is required." >&2
  exit 1
fi

BUCKET=${BUCKET:-"${PROJECT_ID}-blueprint-scenes"}

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

printf "Deploying Genie Sim local job to GKE...\n"
printf "  Project:    %s\n" "${PROJECT_ID}"
printf "  Cluster:    %s\n" "${GKE_CLUSTER}"
printf "  Zone:       %s\n" "${ZONE}"
printf "  Namespace:  %s\n" "${NAMESPACE}"
printf "  Job name:   %s\n" "${JOB_NAME}"

if ! gcloud container clusters describe "${GKE_CLUSTER}" \
  --zone "${ZONE}" \
  --project "${PROJECT_ID}" >/dev/null 2>&1; then
  echo "GKE cluster \"${GKE_CLUSTER}\" not found in zone \"${ZONE}\" for project \"${PROJECT_ID}\"." >&2
  echo "Confirm the cluster name and zone, or set GKE_CLUSTER/ZONE before retrying." >&2
  exit 1
fi

gcloud container clusters get-credentials "${GKE_CLUSTER}" \
  --zone "${ZONE}" \
  --project "${PROJECT_ID}"

if ! kubectl get namespace "${NAMESPACE}" >/dev/null 2>&1; then
  echo "Namespace \"${NAMESPACE}\" was not found in cluster \"${GKE_CLUSTER}\"." >&2
  echo "Create the namespace or set NAMESPACE to an existing namespace before retrying." >&2
  exit 1
fi

if [[ "$(kubectl auth can-i apply -n "${NAMESPACE}" --all 2>/dev/null)" != "yes" ]]; then
  echo "Current kubectl context does not have permission to apply resources in namespace \"${NAMESPACE}\"." >&2
  echo "Grant apply permissions or switch to a context with sufficient access." >&2
  exit 1
fi

kubectl apply -n "${NAMESPACE}" -f "${REPO_ROOT}/k8s/namespace-setup.yaml"

envsubst < "${REPO_ROOT}/k8s/genie-sim-local-job.yaml" | kubectl apply -n "${NAMESPACE}" -f -

printf "Genie Sim local job submitted: %s\n" "${JOB_NAME}"
