#!/bin/bash
# =============================================================================
# Setup EventArc trigger for source-orchestrator (text-first scene requests)
# =============================================================================
#
# Usage:
#   ./setup-source-orchestrator-trigger.sh <project_id> <bucket_name> [workflow_region] [text_gen_runtime]
#
# Trigger target:
#   scenes/{scene_id}/prompts/scene_request.json
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ID=${1:-$(gcloud config get-value project)}
BUCKET=${2:-"${PROJECT_ID}-blueprint-scenes"}
PRIMARY_BUCKET=${PRIMARY_BUCKET:-"${BUCKET}"}
WORKFLOW_REGION=${3:-"us-central1"}
TEXT_GEN_RUNTIME=${4:-${TEXT_GEN_RUNTIME:-"vm"}}
TEXT_SCENE_GEN_JOB_NAME=${TEXT_SCENE_GEN_JOB_NAME:-"text-scene-gen-job"}
TEXT_SCENE_ADAPTER_JOB_NAME=${TEXT_SCENE_ADAPTER_JOB_NAME:-"text-scene-adapter-job"}
DEFAULT_SOURCE_MODE=${DEFAULT_SOURCE_MODE:-"text"}
TEXT_BACKEND_DEFAULT=${TEXT_BACKEND_DEFAULT:-"hybrid_serial"}
TEXT_BACKEND_ALLOWLIST=${TEXT_BACKEND_ALLOWLIST:-"scenesmith,sage,hybrid_serial"}
TEXT_GEN_STANDARD_PROFILE=${TEXT_GEN_STANDARD_PROFILE:-"standard_v1"}
TEXT_GEN_PREMIUM_PROFILE=${TEXT_GEN_PREMIUM_PROFILE:-"premium_v1"}
TEXT_GEN_USE_LLM=${TEXT_GEN_USE_LLM:-"true"}
TEXT_GEN_LLM_MAX_ATTEMPTS=${TEXT_GEN_LLM_MAX_ATTEMPTS:-"3"}
TEXT_GEN_LLM_RETRY_BACKOFF_SECONDS=${TEXT_GEN_LLM_RETRY_BACKOFF_SECONDS:-"2"}
TEXT_ASSET_RETRIEVAL_ENABLED=${TEXT_ASSET_RETRIEVAL_ENABLED:-"true"}
TEXT_ASSET_LIBRARY_PREFIXES=${TEXT_ASSET_LIBRARY_PREFIXES:-"scenes"}
TEXT_ASSET_LIBRARY_MAX_FILES=${TEXT_ASSET_LIBRARY_MAX_FILES:-"2500"}
TEXT_ASSET_LIBRARY_MIN_SCORE=${TEXT_ASSET_LIBRARY_MIN_SCORE:-"0.25"}
TEXT_ASSET_RETRIEVAL_MODE=${TEXT_ASSET_RETRIEVAL_MODE:-"ann_shadow"}
TEXT_ASSET_ANN_ENABLED=${TEXT_ASSET_ANN_ENABLED:-"true"}
TEXT_ASSET_ANN_TOP_K=${TEXT_ASSET_ANN_TOP_K:-"40"}
TEXT_ASSET_ANN_MIN_SCORE=${TEXT_ASSET_ANN_MIN_SCORE:-"0.28"}
TEXT_ASSET_ANN_MAX_RERANK=${TEXT_ASSET_ANN_MAX_RERANK:-"20"}
TEXT_ASSET_ANN_NAMESPACE=${TEXT_ASSET_ANN_NAMESPACE:-"assets-v1"}
TEXT_ASSET_LEXICAL_FALLBACK_ENABLED=${TEXT_ASSET_LEXICAL_FALLBACK_ENABLED:-"true"}
TEXT_ASSET_ROLLOUT_STATE_PREFIX=${TEXT_ASSET_ROLLOUT_STATE_PREFIX:-"automation/asset_retrieval_rollout"}
TEXT_ASSET_ROLLOUT_MIN_DECISIONS=${TEXT_ASSET_ROLLOUT_MIN_DECISIONS:-"500"}
TEXT_ASSET_ROLLOUT_MIN_HIT_RATE=${TEXT_ASSET_ROLLOUT_MIN_HIT_RATE:-"0.95"}
TEXT_ASSET_ROLLOUT_MAX_ERROR_RATE=${TEXT_ASSET_ROLLOUT_MAX_ERROR_RATE:-"0.01"}
TEXT_ASSET_ROLLOUT_MAX_P95_MS=${TEXT_ASSET_ROLLOUT_MAX_P95_MS:-"400"}
TEXT_ASSET_EMBEDDING_QUEUE_PREFIX=${TEXT_ASSET_EMBEDDING_QUEUE_PREFIX:-"automation/asset_embedding/queue"}
TEXT_ASSET_EMBEDDING_PROCESSED_PREFIX=${TEXT_ASSET_EMBEDDING_PROCESSED_PREFIX:-"automation/asset_embedding/processed"}
TEXT_ASSET_EMBEDDING_FAILED_PREFIX=${TEXT_ASSET_EMBEDDING_FAILED_PREFIX:-"automation/asset_embedding/failed"}
TEXT_ASSET_EMBEDDING_MODEL=${TEXT_ASSET_EMBEDDING_MODEL:-"text-embedding-3-small"}
VECTOR_STORE_PROVIDER=${VECTOR_STORE_PROVIDER:-"vertex"}
VECTOR_STORE_PROJECT_ID=${VECTOR_STORE_PROJECT_ID:-"${PROJECT_ID}"}
VECTOR_STORE_LOCATION=${VECTOR_STORE_LOCATION:-"${WORKFLOW_REGION}"}
VECTOR_STORE_NAMESPACE=${VECTOR_STORE_NAMESPACE:-"assets-v1"}
VECTOR_STORE_DIMENSION=${VECTOR_STORE_DIMENSION:-"1536"}
VERTEX_INDEX_ENDPOINT=${VERTEX_INDEX_ENDPOINT:-""}
VERTEX_DEPLOYED_INDEX_ID=${VERTEX_DEPLOYED_INDEX_ID:-""}
TEXT_ASSET_CATALOG_ENABLED=${TEXT_ASSET_CATALOG_ENABLED:-"true"}
TEXT_ASSET_REPLICATION_ENABLED=${TEXT_ASSET_REPLICATION_ENABLED:-"true"}
TEXT_ASSET_REPLICATION_QUEUE_PREFIX=${TEXT_ASSET_REPLICATION_QUEUE_PREFIX:-"automation/asset_replication/queue"}
TEXT_ASSET_REPLICATION_TARGET=${TEXT_ASSET_REPLICATION_TARGET:-"backblaze_b2"}
TEXT_ASSET_REPLICATION_TARGET_PREFIX=${TEXT_ASSET_REPLICATION_TARGET_PREFIX:-"assets"}
TEXT_ASSET_GENERATION_ENABLED=${TEXT_ASSET_GENERATION_ENABLED:-"true"}
TEXT_ASSET_GENERATION_PROVIDER=${TEXT_ASSET_GENERATION_PROVIDER:-"sam3d"}
TEXT_ASSET_GENERATION_PROVIDER_CHAIN=${TEXT_ASSET_GENERATION_PROVIDER_CHAIN:-"sam3d,hunyuan3d"}
TEXT_ASSET_GENERATED_CACHE_ENABLED=${TEXT_ASSET_GENERATED_CACHE_ENABLED:-"true"}
TEXT_ASSET_GENERATED_CACHE_PREFIX=${TEXT_ASSET_GENERATED_CACHE_PREFIX:-"asset-library/generated-text"}
TEXT_SAM3D_API_HOST=${TEXT_SAM3D_API_HOST:-""}
TEXT_SAM3D_TEXT_ENDPOINTS=${TEXT_SAM3D_TEXT_ENDPOINTS:-"/openapi/v1/text-to-3d,/v1/text-to-3d"}
TEXT_SAM3D_TIMEOUT_SECONDS=${TEXT_SAM3D_TIMEOUT_SECONDS:-"1800"}
TEXT_SAM3D_POLL_SECONDS=${TEXT_SAM3D_POLL_SECONDS:-"10"}
TEXT_HUNYUAN_API_HOST=${TEXT_HUNYUAN_API_HOST:-""}
TEXT_HUNYUAN_TEXT_ENDPOINTS=${TEXT_HUNYUAN_TEXT_ENDPOINTS:-"/openapi/v1/text-to-3d,/v1/text-to-3d"}
TEXT_HUNYUAN_TIMEOUT_SECONDS=${TEXT_HUNYUAN_TIMEOUT_SECONDS:-"1800"}
TEXT_HUNYUAN_POLL_SECONDS=${TEXT_HUNYUAN_POLL_SECONDS:-"10"}
TEXT_GEN_MAX_SEEDS=${TEXT_GEN_MAX_SEEDS:-"16"}
ARENA_EXPORT_REQUIRED=${ARENA_EXPORT_REQUIRED:-"true"}
USE_GENIESIM=${USE_GENIESIM:-"true"}
TEXT_GEN_VM_NAME=${TEXT_GEN_VM_NAME:-"isaac-sim-ubuntu-b"}
TEXT_GEN_VM_ZONE=${TEXT_GEN_VM_ZONE:-"us-east1-b"}
TEXT_GEN_VM_REPO_DIR=${TEXT_GEN_VM_REPO_DIR:-"~/BlueprintPipeline"}
TEXT_GEN_VM_TIMEOUT_SECONDS=${TEXT_GEN_VM_TIMEOUT_SECONDS:-"2400"}
SAGE_RUNTIME_MODE=${SAGE_RUNTIME_MODE:-"cloudrun"}
SAGE_SERVER_URL=${SAGE_SERVER_URL:-""}
SAGE_TIMEOUT_SECONDS=${SAGE_TIMEOUT_SECONDS:-"900"}
SCENESMITH_RUNTIME_MODE=${SCENESMITH_RUNTIME_MODE:-"cloudrun"}
SCENESMITH_SERVER_URL=${SCENESMITH_SERVER_URL:-""}
SCENESMITH_TIMEOUT_SECONDS=${SCENESMITH_TIMEOUT_SECONDS:-"1800"}
SCENESMITH_LIVE_REQUIRED=${SCENESMITH_LIVE_REQUIRED:-"true"}
SAGE_LIVE_REQUIRED=${SAGE_LIVE_REQUIRED:-"true"}
TEXT_ENFORCE_LIVE_BACKENDS=${TEXT_ENFORCE_LIVE_BACKENDS:-"false"}
TEXT_SAGE_ACTION_DEMO_ENABLED=${TEXT_SAGE_ACTION_DEMO_ENABLED:-"false"}
RUNPOD_API_KEY=${RUNPOD_API_KEY:-""}
RUNPOD_CLOUD_TYPE=${RUNPOD_CLOUD_TYPE:-"COMMUNITY"}
RUNPOD_GPU_TYPE=${RUNPOD_GPU_TYPE:-"NVIDIA L40S"}
RUNPOD_IMAGE=${RUNPOD_IMAGE:-"runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"}
RUNPOD_VOLUME_GB=${RUNPOD_VOLUME_GB:-"80"}
RUNPOD_CONTAINER_DISK_GB=${RUNPOD_CONTAINER_DISK_GB:-"40"}
RUNPOD_MIN_VCPU=${RUNPOD_MIN_VCPU:-"8"}
RUNPOD_MIN_MEMORY_GB=${RUNPOD_MIN_MEMORY_GB:-"30"}
RUNPOD_POD_NAME_PREFIX=${RUNPOD_POD_NAME_PREFIX:-"bp-textgen"}
RUNPOD_TEXT_STAGE_TIMEOUT_SECONDS=${RUNPOD_TEXT_STAGE_TIMEOUT_SECONDS:-"10800"}
RUNPOD_BOOT_TIMEOUT_SECONDS=${RUNPOD_BOOT_TIMEOUT_SECONDS:-"900"}
RUNPOD_SSH_READY_TIMEOUT_SECONDS=${RUNPOD_SSH_READY_TIMEOUT_SECONDS:-"180"}
RUNPOD_REPO_DIR=${RUNPOD_REPO_DIR:-"/workspace/BlueprintPipeline"}
RUNPOD_STAGE1_PYTHON_BIN=${RUNPOD_STAGE1_PYTHON_BIN:-"python3"}
RUNPOD_SERVICE_PYTHON_BIN=${RUNPOD_SERVICE_PYTHON_BIN:-"python3"}
RUNPOD_SCENESMITH_REPO_DIR=${RUNPOD_SCENESMITH_REPO_DIR:-"/workspace/scenesmith"}
RUNPOD_SCENESMITH_PYTHON_BIN=${RUNPOD_SCENESMITH_PYTHON_BIN:-"/workspace/scenesmith/.venv/bin/python"}
RUNPOD_BOOTSTRAP_COMMAND=${RUNPOD_BOOTSTRAP_COMMAND:-""}
RUNPOD_TERMINATE_ON_EXIT=${RUNPOD_TERMINATE_ON_EXIT:-"true"}
RUNPOD_GCP_SERVICE_ACCOUNT_JSON_B64=${RUNPOD_GCP_SERVICE_ACCOUNT_JSON_B64:-""}
OPENAI_API_KEY=${OPENAI_API_KEY:-""}
BP_GOOGLE_API_KEY=${GOOGLE_API_KEY:-""}
GEMINI_API_KEY=${GEMINI_API_KEY:-""}
HF_TOKEN=${HF_TOKEN:-""}
GITHUB_TOKEN=${GITHUB_TOKEN:-""}
SCENESMITH_PAPER_BACKEND=${SCENESMITH_PAPER_BACKEND:-""}
SCENESMITH_PAPER_MODEL=${SCENESMITH_PAPER_MODEL:-""}
SCENESMITH_PAPER_MODEL_CHAIN=${SCENESMITH_PAPER_MODEL_CHAIN:-""}
SCENESMITH_PAPER_TIMEOUT_SECONDS=${SCENESMITH_PAPER_TIMEOUT_SECONDS:-""}
SCENESMITH_PAPER_ALL_SAM3D=${SCENESMITH_PAPER_ALL_SAM3D:-""}
SCENESMITH_PAPER_FORCE_GENERATED_ASSETS=${SCENESMITH_PAPER_FORCE_GENERATED_ASSETS:-""}
SCENESMITH_PAPER_DISABLE_ARTICULATED_STRATEGY=${SCENESMITH_PAPER_DISABLE_ARTICULATED_STRATEGY:-""}
SCENESMITH_PAPER_IMAGE_BACKEND=${SCENESMITH_PAPER_IMAGE_BACKEND:-""}
SCENESMITH_PAPER_ENABLE_GEMINI_CONTEXT_IMAGE=${SCENESMITH_PAPER_ENABLE_GEMINI_CONTEXT_IMAGE:-""}
SCENESMITH_PAPER_ENABLE_FURNITURE_CONTEXT_IMAGE=${SCENESMITH_PAPER_ENABLE_FURNITURE_CONTEXT_IMAGE:-""}
SCENESMITH_PAPER_EXTRA_OVERRIDES=${SCENESMITH_PAPER_EXTRA_OVERRIDES:-""}

# Runtime-aware defaults. Explicit env vars still win.
if [[ "${TEXT_GEN_RUNTIME}" == "vm" ]]; then
    if [[ -z "${SCENESMITH_SERVER_URL}" ]]; then
        SCENESMITH_SERVER_URL="http://127.0.0.1:8081/v1/generate"
    fi
    if [[ -z "${SAGE_SERVER_URL}" ]]; then
        SAGE_SERVER_URL="http://127.0.0.1:8082/v1/refine"
    fi
    if [[ "${SCENESMITH_RUNTIME_MODE}" == "cloudrun" ]]; then
        SCENESMITH_RUNTIME_MODE="vm"
    fi
    if [[ "${SAGE_RUNTIME_MODE}" == "cloudrun" ]]; then
        SAGE_RUNTIME_MODE="vm"
    fi
fi

# Eventarc trigger location must match bucket location (appspot buckets are multi-region "US")
TRIGGER_LOCATION="us"

TRIGGER_NAME="scene-request-source-orchestrator-trigger"
WORKFLOW_NAME="source-orchestrator"
WORKFLOW_SA="workflow-invoker"

echo -e "${GREEN}=== Source Orchestrator Trigger Setup ===${NC}"
echo "Project:           ${PROJECT_ID}"
echo "Bucket:            ${BUCKET}"
echo "Trigger location:  ${TRIGGER_LOCATION}"
echo "Workflow region:   ${WORKFLOW_REGION}"
echo "Text runtime:      ${TEXT_GEN_RUNTIME}"
echo "Text backend:      ${TEXT_BACKEND_DEFAULT}"
echo ""

echo -e "${BLUE}Step 1: Enabling required APIs...${NC}"
gcloud services enable \
    eventarc.googleapis.com \
    workflows.googleapis.com \
    workflowexecutions.googleapis.com \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    storage.googleapis.com \
    compute.googleapis.com \
    --project=${PROJECT_ID}
echo -e "${GREEN}APIs enabled${NC}"
echo ""

echo -e "${BLUE}Step 2: Setting up service account...${NC}"
SA_EMAIL="${WORKFLOW_SA}@${PROJECT_ID}.iam.gserviceaccount.com"
PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
CLOUD_BUILD_SA="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"

if ! gcloud iam service-accounts describe ${SA_EMAIL} --project=${PROJECT_ID} 2>/dev/null; then
    echo -e "${YELLOW}Creating service account ${WORKFLOW_SA}...${NC}"
    gcloud iam service-accounts create ${WORKFLOW_SA} \
        --display-name="Workflow Invoker SA" \
        --project=${PROJECT_ID}
    sleep 5
else
    echo "Service account ${WORKFLOW_SA} already exists"
fi

echo "Granting permissions..."
for ROLE in \
    "roles/workflows.invoker" \
    "roles/run.invoker" \
    "roles/storage.objectAdmin" \
    "roles/logging.logWriter" \
    "roles/compute.instanceAdmin.v1" \
    "roles/cloudbuild.builds.editor" \
    "roles/eventarc.eventReceiver"; do
    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
        --member="serviceAccount:${SA_EMAIL}" \
        --role="${ROLE}" \
        --quiet 2>&1 | tail -1
done

for ROLE in \
    "roles/compute.instanceAdmin.v1" \
    "roles/compute.osAdminLogin" \
    "roles/logging.logWriter"; do
    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
        --member="serviceAccount:${CLOUD_BUILD_SA}" \
        --role="${ROLE}" \
        --quiet 2>&1 | tail -1
done

GCS_SA="service-${PROJECT_NUMBER}@gs-project-accounts.iam.gserviceaccount.com"
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${GCS_SA}" \
    --role="roles/pubsub.publisher" \
    --quiet 2>&1 | tail -1

echo -e "${GREEN}Permissions granted${NC}"
echo ""

echo -e "${BLUE}Step 3: Deploying source orchestrator workflow...${NC}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Cloud Workflows has a 20 env-var limit. We pack non-default RunPod and
# SceneSmith-paper settings into a single JSON blob (RUNPOD_EXTRA_JSON and
# SCENESMITH_PAPER_JSON) that the orchestrator unpacks at runtime.
# Only keys whose values differ from hardcoded defaults need to be in the blob.

_build_json_blob() {
    # Usage: _build_json_blob KEY1 KEY2 ...
    # Emits compact JSON object of non-empty env vars.
    local pairs=()
    for key in "$@"; do
        local val="${!key:-}"
        if [[ -n "${val}" ]]; then
            pairs+=("\"${key}\": $(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "${val}")")
        fi
    done
    if [[ ${#pairs[@]} -eq 0 ]]; then
        echo "{}"
    else
        local IFS=","
        echo "{${pairs[*]}}"
    fi
}

# Pack RunPod knobs (defaults are hardcoded in orchestrator YAML)
RUNPOD_EXTRA_JSON=$(_build_json_blob \
    RUNPOD_CLOUD_TYPE RUNPOD_GPU_TYPE RUNPOD_IMAGE RUNPOD_VOLUME_GB \
    RUNPOD_CONTAINER_DISK_GB RUNPOD_MIN_VCPU RUNPOD_MIN_MEMORY_GB \
    RUNPOD_POD_NAME_PREFIX RUNPOD_TEXT_STAGE_TIMEOUT_SECONDS \
    RUNPOD_BOOT_TIMEOUT_SECONDS RUNPOD_SSH_READY_TIMEOUT_SECONDS \
    RUNPOD_REPO_DIR RUNPOD_STAGE1_PYTHON_BIN RUNPOD_SERVICE_PYTHON_BIN \
    RUNPOD_SCENESMITH_REPO_DIR RUNPOD_SCENESMITH_PYTHON_BIN \
    RUNPOD_BOOTSTRAP_COMMAND RUNPOD_TERMINATE_ON_EXIT \
    RUNPOD_GCP_SERVICE_ACCOUNT_JSON_B64 \
    HF_TOKEN GITHUB_TOKEN \
)

# Pack SceneSmith paper flags
SCENESMITH_PAPER_JSON=$(_build_json_blob \
    SCENESMITH_PAPER_BACKEND SCENESMITH_PAPER_MODEL SCENESMITH_PAPER_MODEL_CHAIN \
    SCENESMITH_PAPER_TIMEOUT_SECONDS SCENESMITH_PAPER_ALL_SAM3D \
    SCENESMITH_PAPER_FORCE_GENERATED_ASSETS \
    SCENESMITH_PAPER_DISABLE_ARTICULATED_STRATEGY \
    SCENESMITH_PAPER_IMAGE_BACKEND SCENESMITH_PAPER_ENABLE_GEMINI_CONTEXT_IMAGE \
    SCENESMITH_PAPER_ENABLE_FURNITURE_CONTEXT_IMAGE SCENESMITH_PAPER_EXTRA_OVERRIDES \
)

# Pack text backend runtime knobs so we can stay under the 20-env-var limit.
TEXT_BACKEND_RUNTIME_JSON=$(_build_json_blob \
    SAGE_RUNTIME_MODE SAGE_SERVER_URL SAGE_TIMEOUT_SECONDS \
    SCENESMITH_RUNTIME_MODE SCENESMITH_SERVER_URL SCENESMITH_TIMEOUT_SECONDS \
    SCENESMITH_LIVE_REQUIRED SAGE_LIVE_REQUIRED \
    TEXT_ENFORCE_LIVE_BACKENDS TEXT_SAGE_ACTION_DEMO_ENABLED \
)

# These keys go as individual env vars (within the limit)
WORKFLOW_ENV_KEYS=(
    DEFAULT_SOURCE_MODE
    TEXT_BACKEND_DEFAULT
    TEXT_BACKEND_ALLOWLIST
    TEXT_GEN_RUNTIME
    PRIMARY_BUCKET
    TEXT_GEN_MAX_SEEDS
    ARENA_EXPORT_REQUIRED
    USE_GENIESIM
    TEXT_GEN_VM_NAME
    TEXT_GEN_VM_ZONE
    RUNPOD_API_KEY
    OPENAI_API_KEY
    BP_GOOGLE_API_KEY
    GEMINI_API_KEY
    RUNPOD_EXTRA_JSON
    SCENESMITH_PAPER_JSON
    TEXT_BACKEND_RUNTIME_JSON
)

WORKFLOW_ENV_PAIRS=()
for key in "${WORKFLOW_ENV_KEYS[@]}"; do
    WORKFLOW_ENV_PAIRS+=("${key}=${!key}")
done
WORKFLOW_ENV_FILE="$(mktemp)"
trap 'rm -f "${WORKFLOW_ENV_FILE}"' EXIT
for pair in "${WORKFLOW_ENV_PAIRS[@]}"; do
    key="${pair%%=*}"
    value="${pair#*=}"
    escaped_value="${value//\'/\'\'}"
    printf "%s: '%s'\n" "${key}" "${escaped_value}" >> "${WORKFLOW_ENV_FILE}"
done

gcloud workflows deploy ${WORKFLOW_NAME} \
    --location=${WORKFLOW_REGION} \
    --source="${SCRIPT_DIR}/source-orchestrator.yaml" \
    --env-vars-file="${WORKFLOW_ENV_FILE}" \
    --service-account=${SA_EMAIL} \
    --project=${PROJECT_ID}
echo -e "${GREEN}Workflow deployed${NC}"
echo ""

echo -e "${BLUE}Step 4: Recreating trigger if needed...${NC}"
if gcloud eventarc triggers describe ${TRIGGER_NAME} --location=${TRIGGER_LOCATION} --project=${PROJECT_ID} 2>/dev/null; then
    echo -e "${YELLOW}Trigger ${TRIGGER_NAME} exists; deleting and recreating...${NC}"
    gcloud eventarc triggers delete ${TRIGGER_NAME} \
        --location=${TRIGGER_LOCATION} \
        --project=${PROJECT_ID} \
        --quiet
    sleep 5
fi

echo -e "${BLUE}Step 5: Creating EventArc trigger...${NC}"
gcloud eventarc triggers create ${TRIGGER_NAME} \
    --location=${TRIGGER_LOCATION} \
    --project=${PROJECT_ID} \
    --service-account="${SA_EMAIL}" \
    --destination-workflow=${WORKFLOW_NAME} \
    --destination-workflow-location=${WORKFLOW_REGION} \
    --event-filters="type=google.cloud.storage.object.v1.finalized" \
    --event-filters="bucket=${BUCKET}" \
    --event-data-content-type="application/json"
echo -e "${GREEN}Trigger created${NC}"
echo ""

echo -e "${BLUE}Step 6: Verifying trigger...${NC}"
gcloud eventarc triggers describe ${TRIGGER_NAME} \
    --location=${TRIGGER_LOCATION} \
    --project=${PROJECT_ID} \
    --format="table(displayName, destination.workflow, eventFilters)"
echo ""

echo -e "${GREEN}=== Setup Complete ===${NC}"
echo "Workflow: ${WORKFLOW_NAME}"
echo "Trigger:  ${TRIGGER_NAME}"
echo ""
echo "The workflow triggers on bucket object finalize events and filters internally for:"
echo "  scenes/{scene_id}/prompts/scene_request.json"
echo ""
echo "To test:"
echo "  gsutil cp request.json gs://${BUCKET}/scenes/test_scene/prompts/scene_request.json"
