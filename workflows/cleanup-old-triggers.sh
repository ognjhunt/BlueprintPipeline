#!/bin/bash
# =============================================================================
# Cleanup old marker-chain Eventarc triggers
# =============================================================================
#
# Deletes the Eventarc triggers that are replaced by the single
# image-to-scene-orchestrator workflow.
#
# SAFE TO RUN: Only deletes triggers that the orchestrator replaces.
# Keeps independent triggers (objects, scale, layout, etc.) intact.
#
# Usage:
#   ./cleanup-old-triggers.sh [project_id] [trigger_location]
#
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ID=${1:-$(gcloud config get-value project)}
TRIGGER_LOCATION=${2:-"us"}

echo -e "${GREEN}=== Cleanup Old Pipeline Triggers ===${NC}"
echo "Project:  ${PROJECT_ID}"
echo "Location: ${TRIGGER_LOCATION}"
echo ""

# Triggers replaced by the orchestrator
TRIGGERS_TO_DELETE=(
    "image-upload-pipeline-trigger"
    "usd-assembly-trigger"
    "genie-sim-export-trigger"
)

# Stale/orphaned triggers (no active bucket notifications)
STALE_TRIGGERS=(
    "extractframes-951631"
    "sfm-nurec-trigger"
    "mesh-from-da3-trigger"
    "da3-frames-index-trigger"
)

echo -e "${BLUE}Deleting triggers replaced by orchestrator...${NC}"
for TRIGGER in "${TRIGGERS_TO_DELETE[@]}"; do
    if gcloud eventarc triggers describe "${TRIGGER}" \
        --location=${TRIGGER_LOCATION} --project=${PROJECT_ID} 2>/dev/null; then
        echo -e "${YELLOW}Deleting ${TRIGGER}...${NC}"
        gcloud eventarc triggers delete "${TRIGGER}" \
            --location=${TRIGGER_LOCATION} \
            --project=${PROJECT_ID} \
            --quiet
        echo -e "${GREEN}Deleted ${TRIGGER}${NC}"
    else
        echo "  ${TRIGGER} — not found (already deleted)"
    fi
done

echo ""
echo -e "${BLUE}Deleting stale/orphaned triggers...${NC}"
for TRIGGER in "${STALE_TRIGGERS[@]}"; do
    if gcloud eventarc triggers describe "${TRIGGER}" \
        --location=${TRIGGER_LOCATION} --project=${PROJECT_ID} 2>/dev/null; then
        echo -e "${YELLOW}Deleting ${TRIGGER}...${NC}"
        gcloud eventarc triggers delete "${TRIGGER}" \
            --location=${TRIGGER_LOCATION} \
            --project=${PROJECT_ID} \
            --quiet
        echo -e "${GREEN}Deleted ${TRIGGER}${NC}"
    else
        echo "  ${TRIGGER} — not found (already deleted)"
    fi
done

echo ""
echo -e "${GREEN}=== Cleanup Complete ===${NC}"
echo ""
echo "Remaining triggers:"
gcloud eventarc triggers list --location=${TRIGGER_LOCATION} \
    --project=${PROJECT_ID} --format="table(name.basename(), destination.workflow.basename())"
echo ""
echo "Remaining bucket notifications:"
gsutil notification list "gs://${PROJECT_ID}.appspot.com" 2>/dev/null | grep -c "notificationConfigs" || echo "0"
echo ""
