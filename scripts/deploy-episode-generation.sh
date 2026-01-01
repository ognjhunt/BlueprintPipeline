#!/bin/bash
# =============================================================================
# Full Deployment Script for Episode Generation Pipeline
# =============================================================================
#
# This script deploys the complete episode generation infrastructure:
#   1. GKE cluster with GPU node pool (if not exists)
#   2. Kubernetes namespace, RBAC, and secrets
#   3. Isaac Sim container image to Artifact Registry
#   4. Episode generation Kubernetes Job/CronJob
#   5. Cloud Workflow for orchestration
#   6. EventArc trigger for automation
#
# Usage:
#   ./scripts/deploy-episode-generation.sh [project_id] [bucket] [region]
#
# Prerequisites:
#   - gcloud CLI authenticated with project owner/editor
#   - Docker installed and authenticated with gcr.io/nvcr.io
#   - kubectl installed
#   - NVIDIA NGC API key (for Isaac Sim image)
#
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
PROJECT_ID=${1:-$(gcloud config get-value project)}
BUCKET=${2:-"${PROJECT_ID}-blueprint-scenes"}
REGION=${3:-"us-central1"}
ZONE="${REGION}-a"

GKE_CLUSTER="blueprint-cluster"
GKE_GPU_POOL="gpu-pool"
NAMESPACE="blueprint"
AR_REPO="blueprint-jobs"
IMAGE_NAME="episode-gen-job"

# Paths (relative to repo root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"

echo -e "${CYAN}"
echo "============================================================================="
echo "  Episode Generation Pipeline - Full Deployment"
echo "============================================================================="
echo -e "${NC}"
echo "Configuration:"
echo "  Project:     ${PROJECT_ID}"
echo "  Bucket:      ${BUCKET}"
echo "  Region:      ${REGION}"
echo "  GKE Cluster: ${GKE_CLUSTER}"
echo "  Namespace:   ${NAMESPACE}"
echo ""

# Prompt for confirmation
read -p "Continue with deployment? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled."
    exit 0
fi

# =============================================================================
# Step 1: Enable APIs
# =============================================================================
echo ""
echo -e "${BLUE}Step 1/8: Enabling required APIs...${NC}"

gcloud services enable \
    container.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    workflows.googleapis.com \
    eventarc.googleapis.com \
    storage.googleapis.com \
    secretmanager.googleapis.com \
    --project=${PROJECT_ID}

echo -e "${GREEN}✓ APIs enabled${NC}"

# =============================================================================
# Step 2: Create Artifact Registry repository
# =============================================================================
echo ""
echo -e "${BLUE}Step 2/8: Setting up Artifact Registry...${NC}"

if ! gcloud artifacts repositories describe ${AR_REPO} \
    --location=${REGION} --project=${PROJECT_ID} 2>/dev/null; then
    echo "Creating Artifact Registry repository..."
    gcloud artifacts repositories create ${AR_REPO} \
        --repository-format=docker \
        --location=${REGION} \
        --description="BlueprintPipeline container images" \
        --project=${PROJECT_ID}
fi

# Configure Docker auth
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

echo -e "${GREEN}✓ Artifact Registry ready${NC}"

# =============================================================================
# Step 3: Create/verify GKE cluster with GPU node pool
# =============================================================================
echo ""
echo -e "${BLUE}Step 3/8: Setting up GKE cluster...${NC}"

# Check if cluster exists
if ! gcloud container clusters describe ${GKE_CLUSTER} \
    --zone=${ZONE} --project=${PROJECT_ID} 2>/dev/null; then
    echo "Creating GKE cluster..."
    gcloud container clusters create ${GKE_CLUSTER} \
        --zone=${ZONE} \
        --project=${PROJECT_ID} \
        --machine-type=e2-standard-4 \
        --num-nodes=1 \
        --enable-autoscaling --min-nodes=0 --max-nodes=3 \
        --workload-pool=${PROJECT_ID}.svc.id.goog \
        --enable-ip-alias \
        --release-channel=regular
else
    echo "Cluster ${GKE_CLUSTER} already exists"
fi

# Check if GPU node pool exists
if ! gcloud container node-pools describe ${GKE_GPU_POOL} \
    --cluster=${GKE_CLUSTER} --zone=${ZONE} --project=${PROJECT_ID} 2>/dev/null; then
    echo "Creating GPU node pool..."
    gcloud container node-pools create ${GKE_GPU_POOL} \
        --cluster=${GKE_CLUSTER} \
        --zone=${ZONE} \
        --project=${PROJECT_ID} \
        --machine-type=n1-standard-8 \
        --accelerator=type=nvidia-tesla-t4,count=1 \
        --num-nodes=0 \
        --enable-autoscaling --min-nodes=0 --max-nodes=4 \
        --disk-size=200GB \
        --disk-type=pd-ssd \
        --node-taints=nvidia.com/gpu=present:NoSchedule
else
    echo "GPU node pool ${GKE_GPU_POOL} already exists"
fi

# Get credentials
gcloud container clusters get-credentials ${GKE_CLUSTER} \
    --zone=${ZONE} --project=${PROJECT_ID}

# Install NVIDIA device plugin (if not present)
if ! kubectl get daemonset -n kube-system nvidia-driver-installer 2>/dev/null; then
    echo "Installing NVIDIA GPU device plugin..."
    kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
fi

echo -e "${GREEN}✓ GKE cluster ready with GPU support${NC}"

# =============================================================================
# Step 4: Create Kubernetes namespace and RBAC
# =============================================================================
echo ""
echo -e "${BLUE}Step 4/8: Setting up Kubernetes namespace and RBAC...${NC}"

kubectl apply -f ${REPO_ROOT}/k8s/namespace-setup.yaml

# Create GCS service account secret (if not exists)
if ! kubectl get secret gcs-service-account -n ${NAMESPACE} 2>/dev/null; then
    echo ""
    echo -e "${YELLOW}GCS service account secret not found.${NC}"
    echo "Please provide the path to your service account JSON key file:"
    read -p "Service account key path: " SA_KEY_PATH

    if [ -f "${SA_KEY_PATH}" ]; then
        kubectl create secret generic gcs-service-account \
            --from-file=key.json=${SA_KEY_PATH} \
            -n ${NAMESPACE}
        echo -e "${GREEN}✓ Secret created${NC}"
    else
        echo -e "${RED}File not found: ${SA_KEY_PATH}${NC}"
        echo "Creating placeholder secret (update later with real credentials)"
        kubectl create secret generic gcs-service-account \
            --from-literal=key.json='{}' \
            -n ${NAMESPACE} || true
    fi
fi

# Create API key secrets (if not exists)
if ! kubectl get secret episode-gen-secrets -n ${NAMESPACE} 2>/dev/null; then
    echo ""
    echo -e "${YELLOW}API key secrets not found.${NC}"
    read -p "Gemini API Key (or press Enter to skip): " GEMINI_KEY
    read -p "OpenAI API Key (or press Enter to skip): " OPENAI_KEY

    kubectl create secret generic episode-gen-secrets \
        --from-literal=GEMINI_API_KEY="${GEMINI_KEY:-placeholder}" \
        --from-literal=OPENAI_API_KEY="${OPENAI_KEY:-placeholder}" \
        -n ${NAMESPACE} || true
fi

# Setup Workload Identity binding
echo "Setting up Workload Identity..."
GCS_SA="blueprint-pipeline@${PROJECT_ID}.iam.gserviceaccount.com"

# Create GCP service account if not exists
if ! gcloud iam service-accounts describe ${GCS_SA} --project=${PROJECT_ID} 2>/dev/null; then
    gcloud iam service-accounts create blueprint-pipeline \
        --display-name="BlueprintPipeline Service Account" \
        --project=${PROJECT_ID}
fi

# Grant storage access
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${GCS_SA}" \
    --role="roles/storage.objectAdmin" \
    --quiet

# Bind Kubernetes SA to GCP SA
gcloud iam service-accounts add-iam-policy-binding ${GCS_SA} \
    --role="roles/iam.workloadIdentityUser" \
    --member="serviceAccount:${PROJECT_ID}.svc.id.goog[${NAMESPACE}/blueprint-pipeline-sa]" \
    --project=${PROJECT_ID} \
    --quiet

echo -e "${GREEN}✓ Kubernetes namespace and RBAC configured${NC}"

# =============================================================================
# Step 5: Build and push Isaac Sim container image
# =============================================================================
echo ""
echo -e "${BLUE}Step 5/8: Building Isaac Sim container image...${NC}"
echo -e "${YELLOW}Note: This may take 20-40 minutes for the first build${NC}"

cd ${REPO_ROOT}

# Use Cloud Build for the heavy lifting
gcloud builds submit \
    --config=episode-generation-job/cloudbuild.yaml \
    --substitutions=_REGION=${REGION},_REPO=${AR_REPO},_GKE_CLUSTER=${GKE_CLUSTER},_GKE_ZONE=${ZONE} \
    --timeout=7200s \
    .

echo -e "${GREEN}✓ Container image built and pushed${NC}"

# =============================================================================
# Step 6: Deploy Kubernetes Job/CronJob
# =============================================================================
echo ""
echo -e "${BLUE}Step 6/8: Deploying Kubernetes Job configuration...${NC}"

# Update image in manifest
IMAGE_URL="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${IMAGE_NAME}:latest"
sed "s|image:.*episode-gen-job.*|image: ${IMAGE_URL}|g" \
    ${REPO_ROOT}/k8s/episode-generation-job.yaml | \
    sed "s|\${PROJECT_ID}|${PROJECT_ID}|g" | \
    kubectl apply -f -

echo -e "${GREEN}✓ Kubernetes Job deployed${NC}"

# =============================================================================
# Step 7: Deploy Cloud Workflow
# =============================================================================
echo ""
echo -e "${BLUE}Step 7/8: Deploying Cloud Workflow...${NC}"

gcloud workflows deploy episode-generation-pipeline \
    --location=${REGION} \
    --source=${REPO_ROOT}/workflows/episode-generation-pipeline.yaml \
    --description="Episode generation pipeline for Isaac Sim (triggered by .usd_complete)" \
    --project=${PROJECT_ID}

echo -e "${GREEN}✓ Cloud Workflow deployed${NC}"

# =============================================================================
# Step 8: Setup EventArc trigger
# =============================================================================
echo ""
echo -e "${BLUE}Step 8/8: Setting up EventArc trigger...${NC}"

chmod +x ${REPO_ROOT}/episode-generation-job/scripts/setup_eventarc_trigger.sh
${REPO_ROOT}/episode-generation-job/scripts/setup_eventarc_trigger.sh \
    ${PROJECT_ID} ${BUCKET} ${REGION}

echo -e "${GREEN}✓ EventArc trigger configured${NC}"

# =============================================================================
# Deployment Complete
# =============================================================================
echo ""
echo -e "${CYAN}"
echo "============================================================================="
echo "  Deployment Complete!"
echo "============================================================================="
echo -e "${NC}"
echo ""
echo "Episode Generation Pipeline is now deployed and configured."
echo ""
echo "Components:"
echo "  ✓ GKE Cluster:     ${GKE_CLUSTER} (with GPU node pool)"
echo "  ✓ Container Image: ${IMAGE_URL}"
echo "  ✓ K8s Namespace:   ${NAMESPACE}"
echo "  ✓ Cloud Workflow:  episode-generation-pipeline"
echo "  ✓ EventArc:        episode-generation-trigger"
echo ""
echo "Automation Flow:"
echo "  1. USD assembly completes → writes .usd_complete marker"
echo "  2. EventArc triggers → episode-generation-pipeline workflow"
echo "  3. Workflow creates → GKE Job with Isaac Sim GPU"
echo "  4. Isaac Sim generates → physics-validated episodes"
echo "  5. Episodes written → gs://${BUCKET}/scenes/{id}/episodes/"
echo "  6. Workflow writes → .episodes_complete marker"
echo ""
echo "Manual Testing:"
echo "  # Trigger for a specific scene:"
echo "  echo '{}' | gsutil cp - gs://${BUCKET}/scenes/test_scene/usd/.usd_complete"
echo ""
echo "  # Or run directly on GKE:"
echo "  kubectl create job episode-test-\$(date +%s) \\"
echo "    --from=cronjob/episode-generation-cronjob \\"
echo "    -n ${NAMESPACE}"
echo ""
echo "Monitoring:"
echo "  kubectl get jobs -n ${NAMESPACE}"
echo "  kubectl logs job/<job-name> -n ${NAMESPACE}"
echo "  gcloud workflows executions list episode-generation-pipeline --location=${REGION}"
echo ""
