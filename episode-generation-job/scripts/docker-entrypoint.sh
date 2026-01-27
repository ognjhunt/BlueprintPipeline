#!/bin/bash
# =============================================================================
# Isaac Sim Episode Generation Entrypoint
# =============================================================================
# This script handles:
# 1. GCS bucket mounting (if BUCKET is set)
# 2. Isaac Sim environment initialization
# 3. Episode generation execution
# 4. Cleanup and exit
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================

# Paths
ISAAC_SIM_PATH="${ISAAC_SIM_PATH:-/isaac-sim}"
APP_DIR="${APP_DIR:-/app}"
GCS_MOUNT="${GCS_MOUNT:-/mnt/gcs}"
LOG_DIR="${LOG_DIR:-/logs}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Logging Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[EPISODE-GEN]${NC} [INFO] $1"
}

log_success() {
    echo -e "${GREEN}[EPISODE-GEN]${NC} [SUCCESS] $1"
}

log_warning() {
    echo -e "${YELLOW}[EPISODE-GEN]${NC} [WARNING] $1"
}

log_error() {
    echo -e "${RED}[EPISODE-GEN]${NC} [ERROR] $1"
}

# =============================================================================
# Cleanup Function
# =============================================================================

cleanup() {
    log_info "Cleaning up..."

    # Unmount GCS if mounted
    if mountpoint -q "${GCS_MOUNT}"; then
        log_info "Unmounting GCS bucket..."
        fusermount -u "${GCS_MOUNT}" 2>/dev/null || true
    fi

    # Kill any background processes
    jobs -p | xargs -r kill 2>/dev/null || true

    log_info "Cleanup complete"
}

trap cleanup EXIT

# =============================================================================
# Environment Validation
# =============================================================================

validate_environment() {
    log_info "Validating environment..."

    # Check Isaac Sim installation
    if [ ! -f "${ISAAC_SIM_PATH}/python.sh" ]; then
        log_error "Isaac Sim not found at ${ISAAC_SIM_PATH}"
        exit 1
    fi

    # Check for NVIDIA GPU
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found. GPU required for Isaac Sim."
        exit 1
    fi

    # Display GPU info
    log_info "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1

    # Check required environment variables
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

    # Create mount point if needed
    mkdir -p "${GCS_MOUNT}"

    # Check for credentials
    if [ -n "${GOOGLE_APPLICATION_CREDENTIALS}" ]; then
        log_info "Using service account credentials"
    elif [ -n "${CLOUDSDK_AUTH_ACCESS_TOKEN}" ]; then
        log_info "Using access token authentication"
    else
        log_warning "No explicit credentials found, using default"
    fi

    # Mount with gcsfuse
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

    # Set headless mode
    export HEADLESS="${HEADLESS:-1}"

    # Disable GUI/display
    unset DISPLAY

    # Set Nucleus server (if using Omniverse assets)
    if [ -n "${OMNI_SERVER}" ]; then
        export OMNI_SERVER="${OMNI_SERVER}"
    fi

    # Validate Isaac Sim can import required modules
    log_info "Checking Isaac Sim Python environment..."

    ${ISAAC_SIM_PATH}/python.sh -c "
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('isaac-sim-preflight')

logger.info(f'Python: {sys.version}')

# Check core imports
try:
    import omni
    logger.info('- omni: OK')
except ImportError as e:
    logger.error(f'- omni: FAILED ({e})')
    sys.exit(1)

try:
    import omni.isaac.core
    logger.info('- omni.isaac.core: OK')
except ImportError as e:
    logger.error(f'- omni.isaac.core: FAILED ({e})')
    sys.exit(1)

try:
    import omni.physx
    logger.info('- omni.physx: OK')
except ImportError as e:
    logger.error(f'- omni.physx: FAILED ({e})')

try:
    import omni.replicator.core
    logger.info('- omni.replicator.core: OK')
except ImportError as e:
    logger.error(f'- omni.replicator.core: FAILED ({e})')

try:
    from pxr import Usd, UsdGeom
    logger.info('- pxr (USD): OK')
except ImportError as e:
    logger.error(f'- pxr: FAILED ({e})')

logger.info('Isaac Sim environment check complete')
"

    if [ $? -eq 0 ]; then
        log_success "Isaac Sim environment ready"
    else
        log_error "Isaac Sim environment check failed"
        exit 1
    fi
}

# =============================================================================
# Isaac Sim Preflight
# =============================================================================

preflight_isaac_sim() {
    if [ "${PREFLIGHT_ISAAC_SIM:-true}" = "false" ]; then
        log_warning "Isaac Sim preflight disabled (PREFLIGHT_ISAAC_SIM=false)"
        return 0
    fi

    log_info "Running Isaac Sim + Replicator preflight..."

    ${ISAAC_SIM_PATH}/python.sh - <<'PY'
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('isaac-sim-preflight')

def fail(message: str) -> None:
    logger.error(message)
    sys.exit(1)

try:
    import omni  # noqa: F401
except ImportError as exc:
    fail(f"❌ Preflight failed: omni not available ({exc}). Ensure Isaac Sim is installed.")

try:
    import omni.isaac.core  # noqa: F401
except ImportError as exc:
    fail(
        "❌ Preflight failed: omni.isaac.core not available. "
        "Run inside the Isaac Sim Python environment."
    )

try:
    import omni.replicator.core  # noqa: F401
except ImportError as exc:
    fail(
        "❌ Preflight failed: omni.replicator.core not available. "
        "Enable the Replicator extension in Isaac Sim before running."
    )

logger.info("✅ Isaac Sim preflight passed (omni + replicator available)")
PY

    if [ $? -eq 0 ]; then
        log_success "Isaac Sim preflight passed"
    else
        log_error "Isaac Sim preflight failed"
        exit 1
    fi
}

# =============================================================================
# Episode Generation
# =============================================================================

run_episode_generation() {
    log_info "Starting episode generation..."
    log_info "Scene ID: ${SCENE_ID}"
    log_info "Robot: ${ROBOT_TYPE:-franka}"
    log_info "Data Pack: ${DATA_PACK_TIER:-core}"
    log_info "Episodes per variation: ${EPISODES_PER_VARIATION:-10}"

    # Set defaults
    export ASSETS_PREFIX="${ASSETS_PREFIX:-scenes/${SCENE_ID}/assets}"
    export EPISODES_PREFIX="${EPISODES_PREFIX:-scenes/${SCENE_ID}/episodes}"
    export ROBOT_TYPE="${ROBOT_TYPE:-franka}"
    export DATA_PACK_TIER="${DATA_PACK_TIER:-core}"
    export EPISODES_PER_VARIATION="${EPISODES_PER_VARIATION:-10}"
    export FPS="${FPS:-30}"
    export USE_LLM="${USE_LLM:-true}"
    export USE_CPGEN="${USE_CPGEN:-true}"
    export MIN_QUALITY_SCORE="${MIN_QUALITY_SCORE:-0.7}"
    export NUM_CAMERAS="${NUM_CAMERAS:-1}"
    export IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-640,480}"
    export CAPTURE_SENSOR_DATA="${CAPTURE_SENSOR_DATA:-true}"
    export PIPELINE_ENV="${PIPELINE_ENV:-production}"
    export DATA_QUALITY_LEVEL="${DATA_QUALITY_LEVEL:-production}"
    export ISAAC_SIM_REQUIRED="${ISAAC_SIM_REQUIRED:-true}"

    # IMPORTANT: Set USE_MOCK_CAPTURE=false to use real Isaac Sim capture
    export USE_MOCK_CAPTURE="false"
    export SENSOR_CAPTURE_MODE="isaac_sim"

    if [ -z "${SCENE_USD_PATH}" ]; then
        usd_dir="${GCS_MOUNT}/scenes/${SCENE_ID}/usd"
        for ext in usd usda usdz; do
            candidate="${usd_dir}/scene.${ext}"
            if [ -f "${candidate}" ]; then
                export SCENE_USD_PATH="${candidate}"
                break
            fi
        done

        if [ -z "${SCENE_USD_PATH}" ] && [ -d "${usd_dir}" ]; then
            shopt -s nullglob
            for candidate in "${usd_dir}"/*.usd "${usd_dir}"/*.usda "${usd_dir}"/*.usdz; do
                if [ -f "${candidate}" ]; then
                    export SCENE_USD_PATH="${candidate}"
                    break
                fi
            done
            shopt -u nullglob
        fi
    fi

    if [ -n "${SCENE_USD_PATH}" ]; then
        log_info "Using USD scene path: ${SCENE_USD_PATH}"
    elif [ "${DATA_QUALITY_LEVEL}" = "production" ]; then
        log_error "USD scene path is required for production validation (set SCENE_USD_PATH)."
        return 1
    fi

    # Run with Isaac Sim's Python
    cd ${APP_DIR}

    ${ISAAC_SIM_PATH}/python.sh -m episode-generation-job.generate_episodes

    local exit_code=$?

    if [ ${exit_code} -eq 0 ]; then
        log_success "Episode generation completed successfully"
    else
        log_error "Episode generation failed with exit code ${exit_code}"
    fi

    return ${exit_code}
}

# =============================================================================
# Run Validation Only
# =============================================================================

run_validation() {
    log_info "Running validation mode..."

    cd ${APP_DIR}

    ${ISAAC_SIM_PATH}/python.sh -c "
from episode-generation-job.isaac_sim_integration import print_availability_report
print_availability_report()
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
# Custom Script Execution
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
    log_info "Isaac Sim Episode Generation Container"
    log_info "=========================================="
    log_info "Command: ${command}"
    log_info "Scene ID: ${SCENE_ID:-<not set>}"
    log_info "Bucket: ${BUCKET:-<not set>}"
    log_info "=========================================="

    # Validate environment first
    validate_environment

    # Mount GCS if needed
    mount_gcs

    # Initialize Isaac Sim
    init_isaac_sim

    # Fail fast if Replicator or core Isaac Sim modules are missing
    preflight_isaac_sim

    # Execute command
    case "${command}" in
        generate)
            run_episode_generation
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
            log_info "Available commands: generate, validate, shell, custom <script>"
            exit 1
            ;;
    esac
}

# Run main with all arguments
main "$@"
