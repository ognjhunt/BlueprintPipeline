#!/bin/bash
# =============================================================================
# Master EventArc Trigger Setup Script
# =============================================================================
#
# This script sets up all EventArc triggers for the BlueprintPipeline
# workflows in a single command.
#
# It will create triggers for:
#   1. episode-generation-pipeline (manual - uses GKE directly)
#   2. usd-assembly-pipeline
#   3. genie-sim-export-pipeline
#   4. arena-export-pipeline (3 triggers for different sources)
#   5. objects-pipeline
#   6. genie-sim-import-pipeline (webhook-based)
#   7. genie-sim-import-poller (fallback)
#   7. dream2flow-preparation-pipeline (manual)
#   8. dwm-preparation-pipeline (manual)
#
# Usage:
#   ./setup-all-triggers.sh <project_id> [bucket_name] [region]
#
# Prerequisites:
#   - gcloud CLI authenticated
#   - All workflow YAML files deployed
#   - Cloud Run, Workflows, EventArc, and GKE APIs enabled
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

echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     BlueprintPipeline EventArc Trigger Master Setup             ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Configuration:"
echo "  Project ID: ${PROJECT_ID}"
echo "  Bucket: ${BUCKET}"
echo "  Region: ${REGION}"
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

    echo -e "${BLUE}─────────────────────────────────────────────────────────────${NC}"
    echo -e "${BLUE}Setting up: ${description}${NC}"
    echo -e "${BLUE}─────────────────────────────────────────────────────────────${NC}"

    if [ ! -f "${script_name}" ]; then
        echo -e "${RED}ERROR: ${script_name} not found${NC}"
        failed_setups+=("${script_name}")
        setup_status["${script_name}"]="MISSING"
        return 1
    fi

    if bash "${script_name}" "${PROJECT_ID}" "${BUCKET}" "${REGION}" 2>&1; then
        setup_status["${script_name}"]="SUCCESS"
        echo -e "${GREEN}✓ ${description} setup complete${NC}"
    else
        setup_status["${script_name}"]="FAILED"
        failed_setups+=("${script_name}")
        echo -e "${RED}✗ ${description} setup failed${NC}"
    fi

    echo ""
}

# Run all setup scripts in order
run_setup_script "setup-usd-assembly-trigger.sh" "USD Assembly Pipeline"
run_setup_script "setup-genie-sim-export-trigger.sh" "Genie Sim Export Pipeline"
run_setup_script "setup-arena-export-trigger.sh" "Arena Export Pipeline"
run_setup_script "setup-objects-trigger.sh" "Objects Pipeline"
run_setup_script "setup-genie-sim-import-trigger.sh" "Genie Sim Import Pipeline (Webhook)"
run_setup_script "setup-genie-sim-import-poller.sh" "Genie Sim Import Poller (Fallback)"

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
    echo "  1. usd-assembly-trigger      → Trigger on .regen3d_complete"
    echo "  2. genie-sim-export-trigger  → Trigger on .variation_pipeline_complete"
    echo "  3. arena-export-* (3 triggers) → Trigger on .usd_complete, .geniesim_complete, .isaac_lab_complete (ignores .geniesim_submitted)"
    echo "  4. objects-trigger           → Trigger on scene_layout.json uploads"
    echo "  5. genie-sim-import (webhook) → Receives Genie Sim callbacks"
    echo "  6. genie-sim-import-poller → Scheduled fallback poller"
    echo ""
    echo "Manual Setup Still Required:"
    echo "  • Episode Generation: Uses GKE directly (see episode-generation-job/scripts/setup_eventarc_trigger.sh)"
    echo "  • Dream2Flow Preparation: See dream2flow-preparation-pipeline.yaml for trigger spec"
    echo "  • DWM Preparation: See dwm-preparation-pipeline.yaml for trigger spec"
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
