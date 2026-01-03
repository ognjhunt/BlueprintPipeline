#!/bin/bash
# =============================================================================
# Isaac Sim Dream2Flow Preparation Entrypoint
# =============================================================================
# This script handles:
# 1. GCS bucket mounting (if BUCKET is set)
# 2. Isaac Sim environment initialization
# 3. Dream2Flow bundle generation with GPU acceleration
# 4. Cleanup and exit
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================

ISAAC_SIM_PATH="${ISAAC_SIM_PATH:-/isaac-sim}"
APP_DIR="${APP_DIR:-/app}"
GCS_MOUNT="${GCS_MOUNT:-/mnt/gcs}"
LOG_DIR="${LOG_DIR:-/logs}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# =============================================================================
# Logging Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[D2F-ISAACSIM]${NC} [INFO] $1"
}

log_success() {
    echo -e "${GREEN}[D2F-ISAACSIM]${NC} [SUCCESS] $1"
}

log_warning() {
    echo -e "${YELLOW}[D2F-ISAACSIM]${NC} [WARNING] $1"
}

log_error() {
    echo -e "${RED}[D2F-ISAACSIM]${NC} [ERROR] $1"
}

# =============================================================================
# Cleanup Function
# =============================================================================

cleanup() {
    log_info "Cleaning up..."

    if mountpoint -q "${GCS_MOUNT}"; then
        log_info "Unmounting GCS bucket..."
        fusermount -u "${GCS_MOUNT}" 2>/dev/null || true
    fi

    jobs -p | xargs -r kill 2>/dev/null || true
    log_info "Cleanup complete"
}

trap cleanup EXIT

# =============================================================================
# Environment Validation
# =============================================================================

validate_environment() {
    log_info "Validating environment..."

    if [ ! -f "${ISAAC_SIM_PATH}/python.sh" ]; then
        log_error "Isaac Sim not found at ${ISAAC_SIM_PATH}"
        exit 1
    fi

    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found. GPU required for Isaac Sim."
        exit 1
    fi

    log_info "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1

    if [ -z "${SCENE_ID}" ]; then
        log_error "SCENE_ID environment variable is required"
        exit 1
    fi

    log_success "Environment validation passed"
}

# =============================================================================
# GCS Mounting
# =============================================================================

mount_gcs() {
    if [ -z "${BUCKET}" ]; then
        log_info "No BUCKET specified, skipping GCS mount"
        return 0
    fi

    log_info "Mounting GCS bucket: ${BUCKET}"
    mkdir -p "${GCS_MOUNT}"

    if [ -n "${GOOGLE_APPLICATION_CREDENTIALS}" ]; then
        log_info "Using service account credentials"
    fi

    gcsfuse \
        --implicit-dirs \
        --limit-bytes-per-sec -1 \
        --limit-ops-per-sec -1 \
        --stat-cache-ttl 1m \
        --type-cache-ttl 1m \
        --rename-dir-limit 1000000 \
        "${BUCKET}" "${GCS_MOUNT}"

    if mountpoint -q "${GCS_MOUNT}"; then
        log_success "GCS bucket mounted at ${GCS_MOUNT}"
    else
        log_error "Failed to mount GCS bucket"
        exit 1
    fi
}

# =============================================================================
# Isaac Sim Initialization
# =============================================================================

init_isaac_sim() {
    log_info "Initializing Isaac Sim environment..."

    export HEADLESS="${HEADLESS:-1}"
    unset DISPLAY

    log_info "Checking Isaac Sim Python environment..."

    ${ISAAC_SIM_PATH}/python.sh -c "
import sys
print(f'Python: {sys.version}')

try:
    import omni
    print('- omni: OK')
except ImportError as e:
    print(f'- omni: FAILED ({e})')
    sys.exit(1)

try:
    import omni.isaac.core
    print('- omni.isaac.core: OK')
except ImportError as e:
    print(f'- omni.isaac.core: FAILED ({e})')
    sys.exit(1)

try:
    from omni.isaac.kit import SimulationApp
    print('- SimulationApp: OK')
except ImportError as e:
    print(f'- SimulationApp: FAILED ({e})')

try:
    from pxr import Usd, UsdGeom
    print('- pxr (USD): OK')
except ImportError as e:
    print(f'- pxr: FAILED ({e})')

print('Isaac Sim environment check complete')
"

    if [ $? -eq 0 ]; then
        log_success "Isaac Sim environment ready"
    else
        log_error "Isaac Sim environment check failed"
        exit 1
    fi
}

# =============================================================================
# Dream2Flow Generation
# =============================================================================

run_dream2flow_generation() {
    log_info "Starting Dream2Flow bundle generation..."
    log_info "Scene ID: ${SCENE_ID}"
    log_info "Robot: ${ROBOT:-franka_panda}"
    log_info "Tasks: ${NUM_TASKS:-5}"
    log_info "Resolution: ${RESOLUTION_WIDTH:-720}x${RESOLUTION_HEIGHT:-480}"

    # Force Isaac Sim renderer
    export RENDER_BACKEND="isaac_sim"

    cd ${APP_DIR}

    ${ISAAC_SIM_PATH}/python.sh -m dream2flow-preparation-job.entrypoint

    local exit_code=$?

    if [ ${exit_code} -eq 0 ]; then
        log_success "Dream2Flow generation completed successfully"
    else
        log_error "Dream2Flow generation failed with exit code ${exit_code}"
    fi

    return ${exit_code}
}

# =============================================================================
# Inference Mode (for running Dream2Flow models)
# =============================================================================

run_inference() {
    log_info "Running Dream2Flow inference mode..."

    cd ${APP_DIR}

    ${ISAAC_SIM_PATH}/python.sh -m dream2flow-preparation-job.dream2flow_inference_job
}

# =============================================================================
# Validation Mode
# =============================================================================

run_validation() {
    log_info "Running validation mode..."

    cd ${APP_DIR}

    ${ISAAC_SIM_PATH}/python.sh -c "
print('Dream2Flow Isaac Sim Validation')
print('=' * 40)

# Check Isaac Sim rendering
try:
    import omni
    from omni.isaac.kit import SimulationApp
    print('Isaac Sim rendering: AVAILABLE')
except ImportError:
    print('Isaac Sim rendering: NOT AVAILABLE')

# Check vision models (optional)
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
except ImportError:
    print('PyTorch: NOT INSTALLED (will use API endpoints)')

print('=' * 40)
print('Validation complete')
"
}

# =============================================================================
# Interactive Shell
# =============================================================================

run_shell() {
    log_info "Starting interactive Isaac Sim shell..."
    exec ${ISAAC_SIM_PATH}/python.sh
}

# =============================================================================
# Custom Script
# =============================================================================

run_custom() {
    local script="$1"
    shift
    log_info "Running custom script: ${script}"
    cd ${APP_DIR}
    ${ISAAC_SIM_PATH}/python.sh "${script}" "$@"
}

# =============================================================================
# Main Entry Point
# =============================================================================

main() {
    local command="${1:-generate}"

    log_info "=========================================="
    log_info "Isaac Sim Dream2Flow Preparation Container"
    log_info "=========================================="
    log_info "Command: ${command}"
    log_info "Scene ID: ${SCENE_ID:-<not set>}"
    log_info "Bucket: ${BUCKET:-<not set>}"
    log_info "=========================================="

    validate_environment
    mount_gcs
    init_isaac_sim

    case "${command}" in
        generate)
            run_dream2flow_generation
            ;;
        inference)
            run_inference
            ;;
        validate)
            run_validation
            ;;
        shell)
            run_shell
            ;;
        custom)
            shift
            run_custom "$@"
            ;;
        *)
            log_error "Unknown command: ${command}"
            log_info "Available commands: generate, inference, validate, shell, custom <script>"
            exit 1
            ;;
    esac
}

main "$@"
