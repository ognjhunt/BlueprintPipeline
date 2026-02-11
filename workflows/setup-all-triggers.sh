#!/bin/bash
# =============================================================================
# Master EventArc Trigger Setup Script
# =============================================================================
#
# This script sets up all EventArc triggers for the BlueprintPipeline
# workflows in a single command.
#
# It will create triggers for:
#   - source-orchestrator (text-first scene requests)
#   - image path trigger(s), based on IMAGE_PATH_MODE
#     - orchestrator mode: image-to-scene-orchestrator trigger
#     - legacy_chain mode: image-to-scene-pipeline + marker-chain triggers
#   - objects-pipeline
#   - genie-sim-import-poller (fallback)
#
# Usage:
#   ./setup-all-triggers.sh <project_id> [bucket_name] [region] [image_path_mode] [enable_text_autonomy_daily]
# image_path_mode:
#   orchestrator (default) | legacy_chain
# enable_text_autonomy_daily:
#   true | false (default false)
#
# Prerequisites:
#   - gcloud CLI authenticated
#   - All workflow YAML files deployed
#   - Cloud Run, Workflows, EventArc, and GKE services enabled
#
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
PROJECT_ID=${1:-$(gcloud config get-value project)}
BUCKET=${2:-"${PROJECT_ID}-blueprint-scenes"}
REGION=${3:-"us-central1"}
IMAGE_PATH_MODE=${4:-${IMAGE_PATH_MODE:-"orchestrator"}}
ENABLE_TEXT_AUTONOMY_DAILY=${5:-${ENABLE_TEXT_AUTONOMY_DAILY:-"false"}}

echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     BlueprintPipeline EventArc Trigger Master Setup             ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Configuration:"
echo "  Project ID: ${PROJECT_ID}"
echo "  Bucket: ${BUCKET}"
echo "  Region: ${REGION}"
echo "  Image Path Mode: ${IMAGE_PATH_MODE}"
echo "  Enable Text Autonomy Daily: ${ENABLE_TEXT_AUTONOMY_DAILY}"
echo ""

# =============================================================================
# Run individual setup scripts
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Track setup status
declare -A setup_status
failed_setups=()

run_setup_script() {
    local script_name=$1
    local description=$2
    local extra_args=${3:-}

    echo -e "${BLUE}─────────────────────────────────────────────────────────────${NC}"
    echo -e "${BLUE}Setting up: ${description}${NC}"
    echo -e "${BLUE}─────────────────────────────────────────────────────────────${NC}"

    if [ ! -f "${script_name}" ]; then
        echo -e "${RED}ERROR: ${script_name} not found${NC}"
        failed_setups+=("${script_name}")
        setup_status["${script_name}"]="MISSING"
        return 1
    fi

    if bash "${script_name}" "${PROJECT_ID}" "${BUCKET}" "${REGION}" ${extra_args} 2>&1; then
        setup_status["${script_name}"]="SUCCESS"
        echo -e "${GREEN}✓ ${description} setup complete${NC}"
    else
        setup_status["${script_name}"]="FAILED"
        failed_setups+=("${script_name}")
        echo -e "${RED}✗ ${description} setup failed${NC}"
    fi

    echo ""
}

# Run all setup scripts in order.
# IMAGE_PATH_MODE=orchestrator: avoid marker-chain trigger duplication by using orchestrator-only topology.
# IMAGE_PATH_MODE=legacy_chain: keep the legacy marker-chain topology active.
if [ "${IMAGE_PATH_MODE}" = "orchestrator" ]; then
    run_setup_script "setup-orchestrator-trigger.sh" "Image-to-Scene Orchestrator"
    run_setup_script "setup-source-orchestrator-trigger.sh" "Source Orchestrator (Text-First)" "${IMAGE_PATH_MODE}"
else
    run_setup_script "setup-image-trigger.sh" "Image-to-Scene Pipeline"
    run_setup_script "setup-source-orchestrator-trigger.sh" "Source Orchestrator (Text-First)" "${IMAGE_PATH_MODE}"
    run_setup_script "setup-usd-assembly-trigger.sh" "USD Assembly Pipeline"
    run_setup_script "setup-genie-sim-export-trigger.sh" "Genie Sim Export Pipeline"
    run_setup_script "setup-arena-export-trigger.sh" "Arena Export Pipeline"
fi
run_setup_script "setup-objects-trigger.sh" "Objects Pipeline"
run_setup_script "setup-genie-sim-import-poller.sh" "Genie Sim Import Poller (Fallback)"
if [ "${ENABLE_TEXT_AUTONOMY_DAILY}" = "true" ]; then
    run_setup_script "setup-text-autonomy-scheduler.sh" "Text Autonomy Daily Scheduler"
fi

# =============================================================================
# Summary
# =============================================================================

echo -e "${GREEN}═════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}SETUP SUMMARY${NC}"
echo -e "${GREEN}═════════════════════════════════════════════════════════════${NC}"
echo ""

success_count=0
fail_count=0

for script in "${!setup_status[@]}"; do
    status=${setup_status[$script]}
    if [ "$status" = "SUCCESS" ]; then
        echo -e "${GREEN}✓${NC} $script"
        ((success_count++))
    elif [ "$status" = "FAILED" ]; then
        echo -e "${RED}✗${NC} $script"
        ((fail_count++))
    elif [ "$status" = "MISSING" ]; then
        echo -e "${YELLOW}?${NC} $script (not found)"
        ((fail_count++))
    fi
done

echo ""
echo "Summary: ${success_count} succeeded, ${fail_count} failed"
echo ""

if [ ${fail_count} -eq 0 ]; then
    echo -e "${GREEN}═════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}All EventArc triggers have been successfully set up!${NC}"
    echo -e "${GREEN}═════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "Pipeline Triggers Created:"
    if [ "${IMAGE_PATH_MODE}" = "orchestrator" ]; then
        echo "  0. image-upload-orchestrator-trigger → Trigger on scenes/{scene_id}/images/* uploads"
        echo "  1. scene-request-source-orchestrator-trigger → Trigger on scene_request.json uploads"
        echo "  2. objects-trigger           → Trigger on scene_layout.json uploads"
        echo "  3. genie-sim-import-poller  → Scheduled fallback poller"
        if [ "${ENABLE_TEXT_AUTONOMY_DAILY}" = "true" ]; then
            echo "  4. text-autonomy-daily      → Scheduled daily text autonomy workflow"
        fi
        echo ""
        echo "Note: marker-chain triggers (usd/geniesim/arena) were intentionally skipped to avoid duplicate topology."
    else
        echo "  0. image-upload-pipeline-trigger → Trigger on scenes/{scene_id}/images/* uploads"
        echo "  1. scene-request-source-orchestrator-trigger → Trigger on scene_request.json uploads"
        echo "  2. usd-assembly-trigger      → Trigger on .regen3d_complete"
        echo "  3. genie-sim-export-trigger  → Trigger on .variation_pipeline_complete"
        echo "  4. arena-export-* (3 triggers) → Trigger on .usd_complete, .geniesim_complete, .isaac_lab_complete (ignores .geniesim_submitted)"
        echo "  5. objects-trigger           → Trigger on scene_layout.json uploads"
        echo "  6. genie-sim-import-poller  → Scheduled fallback poller"
        if [ "${ENABLE_TEXT_AUTONOMY_DAILY}" = "true" ]; then
            echo "  7. text-autonomy-daily      → Scheduled daily text autonomy workflow"
        fi
    fi
    echo ""
    echo "Optional Additional Setup:"
    echo "  • Dream2Flow Preparation (disabled by default): See dream2flow-preparation-pipeline.yaml for trigger spec"
    echo "  • DWM Preparation (disabled by default): See dwm-preparation-pipeline.yaml for trigger spec"
    echo ""
    echo "To test a trigger manually:"
    echo "  echo '{}' | gsutil cp - gs://${BUCKET}/scenes/test_scene/usd/.regen3d_complete"
    echo ""
    echo "To view all triggers:"
    echo "  gcloud eventarc triggers list --location=${REGION} --project=${PROJECT_ID}"
    echo ""
else
    echo -e "${RED}═════════════════════════════════════════════════════════════${NC}"
    echo -e "${RED}Some setups failed. Please review errors above.${NC}"
    echo -e "${RED}═════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "Failed setups:"
    for failed in "${failed_setups[@]}"; do
        echo "  - $failed"
    done
    echo ""
    exit 1
fi
