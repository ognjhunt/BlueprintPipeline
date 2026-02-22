#!/bin/bash
# =============================================================================
# Master EventArc Trigger Setup Script (Text-Only Stage 1)
# =============================================================================
#
# This script sets up all EventArc triggers for the BlueprintPipeline workflows
# in a single command.
#
# It creates triggers for:
#   - source-orchestrator (scene_request.json)
#   - usd-assembly (Stage 1 completion marker)
#   - genie-sim-export / arena-export downstream stages
#   - objects-pipeline
#   - genie-sim-import-poller (fallback)
#
# Usage:
#   ./setup-all-triggers.sh <project_id> [bucket_name] [region] [enable_text_autonomy_daily] [enable_asset_replication] [enable_asset_embedding]
#
# enable_text_autonomy_daily:
#   true | false (default false)
# enable_asset_replication:
#   true | false (default false)
# enable_asset_embedding:
#   true | false (default false)
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
ENABLE_TEXT_AUTONOMY_DAILY=${4:-${ENABLE_TEXT_AUTONOMY_DAILY:-"false"}}
ENABLE_ASSET_REPLICATION=${5:-${ENABLE_ASSET_REPLICATION:-"false"}}
ENABLE_ASSET_EMBEDDING=${6:-${ENABLE_ASSET_EMBEDDING:-"false"}}

echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     BlueprintPipeline EventArc Trigger Master Setup             ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Configuration:"
echo "  Project ID: ${PROJECT_ID}"
echo "  Bucket: ${BUCKET}"
echo "  Region: ${REGION}"
echo "  Enable Text Autonomy Daily: ${ENABLE_TEXT_AUTONOMY_DAILY}"
echo "  Enable Asset Replication: ${ENABLE_ASSET_REPLICATION}"
echo "  Enable Asset Embedding: ${ENABLE_ASSET_EMBEDDING}"
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

run_setup_script "setup-source-orchestrator-trigger.sh" "Source Orchestrator (Text-First)"
run_setup_script "setup-usd-assembly-trigger.sh" "USD Assembly Pipeline"
run_setup_script "setup-genie-sim-export-trigger.sh" "Genie Sim Export Pipeline"
run_setup_script "setup-arena-export-trigger.sh" "Arena Export Pipeline"
run_setup_script "setup-objects-trigger.sh" "Objects Pipeline"
run_setup_script "setup-genie-sim-import-poller.sh" "Genie Sim Import Poller (Fallback)"
if [ "${ENABLE_TEXT_AUTONOMY_DAILY}" = "true" ]; then
    run_setup_script "setup-text-autonomy-scheduler.sh" "Text Autonomy Daily Scheduler"
fi
if [ "${ENABLE_ASSET_REPLICATION}" = "true" ]; then
    run_setup_script "setup-asset-replication-trigger.sh" "Asset Replication Queue Trigger"
fi
if [ "${ENABLE_ASSET_EMBEDDING}" = "true" ]; then
    run_setup_script "setup-asset-embedding-trigger.sh" "Asset Embedding Queue Trigger"
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
    echo "  0. scene-request-source-orchestrator-trigger → Trigger on scene_request.json uploads"
    echo "  1. usd-assembly-trigger      → Trigger on .stage1_complete"
    echo "  2. genie-sim-export-trigger  → Trigger on .variation_pipeline_complete"
    echo "  3. arena-export-* (3 triggers) → Trigger on .usd_complete, .geniesim_complete, .isaac_lab_complete (ignores .geniesim_submitted)"
    echo "  4. objects-trigger           → Trigger on scene_layout.json uploads"
    echo "  5. genie-sim-import-poller  → Scheduled fallback poller"
    if [ "${ENABLE_TEXT_AUTONOMY_DAILY}" = "true" ]; then
        echo "  6. text-autonomy-daily      → Scheduled daily text autonomy workflow"
    fi
    if [ "${ENABLE_ASSET_REPLICATION}" = "true" ]; then
        echo "  7. asset-replication-trigger → Trigger on automation/asset_replication/queue/*.json"
    fi
    if [ "${ENABLE_ASSET_EMBEDDING}" = "true" ]; then
        echo "  8. asset-embedding-trigger → Trigger on automation/asset_embedding/queue/*.json"
    fi
    echo ""
    echo "Optional Additional Setup:"
    echo "  • Dream2Flow Preparation (disabled by default): See dream2flow-preparation-pipeline.yaml for trigger spec"
    echo "  • DWM Preparation (disabled by default): See dwm-preparation-pipeline.yaml for trigger spec"
    echo ""
    echo "To test a trigger manually:"
    echo "  echo '{}' | gsutil cp - gs://${BUCKET}/scenes/test_scene/assets/.stage1_complete"
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
