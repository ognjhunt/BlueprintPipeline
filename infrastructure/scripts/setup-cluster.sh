#!/bin/bash
# =============================================================================
# BlueprintPipeline - GKE Cluster Setup Script
# =============================================================================
# This script sets up the complete GKE infrastructure for BlueprintPipeline.
#
# Usage:
#   ./setup-cluster.sh --project YOUR_PROJECT_ID [--region us-central1]
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - terraform >= 1.5.0 installed
#   - kubectl installed
#   - Docker installed (for building images)
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TERRAFORM_DIR="${REPO_ROOT}/infrastructure/terraform"

# Default values
PROJECT_ID=""
REGION="us-central1"
ZONE="us-central1-a"
CLUSTER_NAME="blueprint-cluster"
ENVIRONMENT="prod"
GPU_TYPE="nvidia-tesla-t4"
SKIP_TERRAFORM=false
SKIP_BUILD=false
DRY_RUN=false

# =============================================================================
# Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Sets up the GKE infrastructure for BlueprintPipeline.

Required:
  --project PROJECT_ID    GCP project ID

Options:
  --region REGION         GCP region (default: us-central1)
  --zone ZONE            GCP zone (default: us-central1-a)
  --cluster NAME         Cluster name (default: blueprint-cluster)
  --environment ENV      Environment: dev, staging, prod (default: prod)
  --gpu-type TYPE        GPU type (default: nvidia-tesla-t4)
  --skip-terraform       Skip Terraform provisioning
  --skip-build          Skip Docker image builds
  --dry-run             Show what would be done without executing
  --help                 Show this help message

Examples:
  $(basename "$0") --project my-project
  $(basename "$0") --project my-project --region europe-west1 --gpu-type nvidia-l4
  $(basename "$0") --project my-project --skip-build

EOF
    exit 0
}

check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check gcloud
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI not found. Install from: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi

    # Check terraform
    if ! command -v terraform &> /dev/null; then
        log_error "Terraform not found. Install from: https://www.terraform.io/downloads"
        exit 1
    fi

    # Check terraform version
    TF_VERSION=$(terraform version -json | jq -r '.terraform_version' 2>/dev/null || terraform version | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
    if [[ "${TF_VERSION%%.*}" -lt 1 ]]; then
        log_error "Terraform version >= 1.5.0 required. Found: ${TF_VERSION}"
        exit 1
    fi

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Install from: https://kubernetes.io/docs/tasks/tools/"
        exit 1
    fi

    # Check docker (only if not skipping build)
    if [[ "$SKIP_BUILD" == "false" ]] && ! command -v docker &> /dev/null; then
        log_warning "Docker not found. Image builds will be skipped."
        SKIP_BUILD=true
    fi

    # Check gcloud auth
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -1 &> /dev/null; then
        log_error "Not authenticated with gcloud. Run: gcloud auth login"
        exit 1
    fi

    log_success "All prerequisites met"
}

configure_gcloud() {
    log_info "Configuring gcloud..."

    gcloud config set project "${PROJECT_ID}"
    gcloud config set compute/region "${REGION}"
    gcloud config set compute/zone "${ZONE}"

    # Enable required APIs
    log_info "Enabling required GCP APIs..."
    gcloud services enable \
        container.googleapis.com \
        containerregistry.googleapis.com \
        artifactregistry.googleapis.com \
        compute.googleapis.com \
        storage.googleapis.com \
        cloudbuild.googleapis.com \
        workflows.googleapis.com \
        eventarc.googleapis.com \
        run.googleapis.com \
        pubsub.googleapis.com \
        monitoring.googleapis.com \
        logging.googleapis.com \
        --quiet

    log_success "gcloud configured"
}

run_terraform() {
    if [[ "$SKIP_TERRAFORM" == "true" ]]; then
        log_warning "Skipping Terraform provisioning"
        return 0
    fi

    log_info "Running Terraform..."
    cd "${TERRAFORM_DIR}"

    # Initialize Terraform
    log_info "Initializing Terraform..."
    terraform init -upgrade

    # Create tfvars file
    cat > terraform.tfvars << EOF
project_id  = "${PROJECT_ID}"
region      = "${REGION}"
zone        = "${ZONE}"
cluster_name = "${CLUSTER_NAME}"
environment = "${ENVIRONMENT}"
gpu_type    = "${GPU_TYPE}"
EOF

    # Plan
    log_info "Planning Terraform changes..."
    terraform plan -out=tfplan

    if [[ "$DRY_RUN" == "true" ]]; then
        log_warning "Dry run - skipping apply"
        return 0
    fi

    # Apply
    log_info "Applying Terraform changes (this may take 10-15 minutes)..."
    terraform apply tfplan

    # Save outputs
    terraform output -json > terraform-outputs.json

    log_success "Terraform provisioning complete"
    cd "${REPO_ROOT}"
}

configure_kubectl() {
    log_info "Configuring kubectl..."

    gcloud container clusters get-credentials "${CLUSTER_NAME}" \
        --zone "${ZONE}" \
        --project "${PROJECT_ID}"

    # Verify connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Failed to connect to cluster"
        exit 1
    fi

    # Set context name
    kubectl config rename-context \
        "$(kubectl config current-context)" \
        "blueprint-${ENVIRONMENT}" 2>/dev/null || true

    log_success "kubectl configured"
}

build_and_push_images() {
    if [[ "$SKIP_BUILD" == "true" ]]; then
        log_warning "Skipping Docker image builds"
        return 0
    fi

    log_info "Building and pushing Docker images..."

    # Configure Docker for Artifact Registry
    gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

    REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/blueprint-jobs"

    # Build episode-generation-job (Isaac Sim)
    log_info "Building episode-generation-job..."
    docker build \
        -f "${REPO_ROOT}/episode-generation-job/Dockerfile.isaacsim" \
        -t "${REGISTRY}/episode-gen-job:latest" \
        "${REPO_ROOT}"

    # Build DWM preparation job
    log_info "Building dwm-preparation-job..."
    docker build \
        -f "${REPO_ROOT}/dwm-preparation-job/Dockerfile.isaacsim" \
        -t "${REGISTRY}/dwm-prep-job:latest" \
        "${REPO_ROOT}"

    # Build Dream2Flow preparation job
    log_info "Building dream2flow-preparation-job..."
    docker build \
        -f "${REPO_ROOT}/dream2flow-preparation-job/Dockerfile.isaacsim" \
        -t "${REGISTRY}/dream2flow-prep-job:latest" \
        "${REPO_ROOT}"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_warning "Dry run - skipping push"
        return 0
    fi

    # Push images
    log_info "Pushing images to Artifact Registry..."
    docker push "${REGISTRY}/episode-gen-job:latest"
    docker push "${REGISTRY}/dwm-prep-job:latest"
    docker push "${REGISTRY}/dream2flow-prep-job:latest"

    log_success "Images built and pushed"
}

deploy_kubernetes_resources() {
    log_info "Deploying Kubernetes resources..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_warning "Dry run - showing what would be deployed"
        kubectl apply -f "${REPO_ROOT}/k8s/namespace-setup.yaml" --dry-run=client
        return 0
    fi

    # Deploy namespace and RBAC
    kubectl apply -f "${REPO_ROOT}/k8s/namespace-setup.yaml"

    # Create secrets (placeholders - should be replaced with real values)
    kubectl create secret generic gcs-service-account \
        --namespace=blueprint \
        --from-file=key.json=/dev/null \
        --dry-run=client -o yaml | kubectl apply -f - 2>/dev/null || true

    kubectl create secret generic api-keys \
        --namespace=blueprint \
        --from-literal=GEMINI_API_KEY="" \
        --from-literal=OPENAI_API_KEY="" \
        --dry-run=client -o yaml | kubectl apply -f - 2>/dev/null || true

    # Apply job templates
    kubectl apply -f "${REPO_ROOT}/k8s/episode-generation-job.yaml"

    # Check if DWM and Dream2Flow manifests exist
    if [[ -f "${REPO_ROOT}/k8s/dwm-preparation-job.yaml" ]]; then
        kubectl apply -f "${REPO_ROOT}/k8s/dwm-preparation-job.yaml"
    fi

    if [[ -f "${REPO_ROOT}/k8s/dream2flow-preparation-job.yaml" ]]; then
        kubectl apply -f "${REPO_ROOT}/k8s/dream2flow-preparation-job.yaml"
    fi

    log_success "Kubernetes resources deployed"
}

setup_monitoring() {
    log_info "Setting up monitoring..."

    # Deploy monitoring dashboards
    if [[ -f "${REPO_ROOT}/infrastructure/monitoring/setup-monitoring.sh" ]]; then
        bash "${REPO_ROOT}/infrastructure/monitoring/setup-monitoring.sh" \
            --project "${PROJECT_ID}" \
            --cluster "${CLUSTER_NAME}" \
            ${DRY_RUN:+--dry-run}
    fi

    log_success "Monitoring setup complete"
}

print_summary() {
    echo ""
    echo "============================================================"
    echo -e "${GREEN}BlueprintPipeline Infrastructure Setup Complete!${NC}"
    echo "============================================================"
    echo ""
    echo "Cluster:     ${CLUSTER_NAME}"
    echo "Project:     ${PROJECT_ID}"
    echo "Region:      ${REGION}"
    echo "Zone:        ${ZONE}"
    echo "Environment: ${ENVIRONMENT}"
    echo "GPU Type:    ${GPU_TYPE}"
    echo ""
    echo "Next steps:"
    echo ""
    echo "  1. Add your GCS service account key:"
    echo "     kubectl create secret generic gcs-service-account \\"
    echo "       --namespace=blueprint \\"
    echo "       --from-file=key.json=PATH_TO_KEY.json"
    echo ""
    echo "  2. Add your API keys:"
    echo "     kubectl create secret generic api-keys \\"
    echo "       --namespace=blueprint \\"
    echo "       --from-literal=GEMINI_API_KEY=your-key"
    echo ""
    echo "  3. Upload a scene to test:"
    echo "     gsutil cp -r test_scene gs://${PROJECT_ID}-blueprint-data/scenes/test/"
    echo ""
    echo "  4. Trigger a pipeline run:"
    echo "     kubectl apply -f k8s/episode-generation-job.yaml"
    echo ""
    echo "  5. Monitor the job:"
    echo "     kubectl logs -f job/episode-generation -n blueprint"
    echo ""
    echo "============================================================"
}

# =============================================================================
# Main
# =============================================================================

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --project)
            PROJECT_ID="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --zone)
            ZONE="$2"
            shift 2
            ;;
        --cluster)
            CLUSTER_NAME="$2"
            shift 2
            ;;
        --environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --gpu-type)
            GPU_TYPE="$2"
            shift 2
            ;;
        --skip-terraform)
            SKIP_TERRAFORM=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            usage
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [[ -z "$PROJECT_ID" ]]; then
    log_error "Project ID is required. Use --project YOUR_PROJECT_ID"
    usage
fi

# Main execution
echo ""
echo "============================================================"
echo "BlueprintPipeline Infrastructure Setup"
echo "============================================================"
echo ""

check_prerequisites
configure_gcloud
run_terraform
configure_kubectl
build_and_push_images
deploy_kubernetes_resources
setup_monitoring
print_summary
