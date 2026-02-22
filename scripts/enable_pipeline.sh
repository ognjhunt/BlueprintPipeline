#!/bin/bash
#
# BlueprintPipeline Scheduler and Trigger Management Script
#
# This script manages the automated triggers for the pipeline:
# - Scene generation scheduler (daily at 8:00 AM)
# - EventArc triggers for pipeline stages
#
# Usage:
#   ./scripts/enable_pipeline.sh enable    # Enable automated pipeline
#   ./scripts/enable_pipeline.sh disable   # Disable automated pipeline
#   ./scripts/enable_pipeline.sh status    # Check current status
#   ./scripts/enable_pipeline.sh setup     # Initial setup of all triggers
#
# Note: Upsell features are processed INLINE during episode generation.
#       The standalone upsell workflow is for manual/retroactive use only.

set -e

# Configuration
PROJECT_ID=${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}
REGION=${REGION:-"us-central1"}
BUCKET=${BUCKET:-"blueprint-scenes"}
WORKFLOW_SA=${WORKFLOW_SA:-"blueprint-workflow-sa"}
SCHEDULER_SA=${SCHEDULER_SA:-"blueprint-scheduler-sa"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================================
# Status Check
# ============================================================================

check_status() {
    echo "=============================================="
    echo "BlueprintPipeline Status Check"
    echo "=============================================="
    echo ""
    echo "Project: $PROJECT_ID"
    echo "Region: $REGION"
    echo "Bucket: $BUCKET"
    echo ""

    # Check scheduler
    echo "--- Scene Generation Scheduler ---"
    SCHEDULER_STATUS=$(gcloud scheduler jobs describe scene-generation-daily \
        --location=$REGION \
        --format='value(state)' 2>/dev/null || echo "NOT_FOUND")

    if [ "$SCHEDULER_STATUS" == "NOT_FOUND" ]; then
        log_warn "Scheduler not found. Run 'setup' to create."
    elif [ "$SCHEDULER_STATUS" == "ENABLED" ]; then
        log_info "Scheduler: ENABLED (daily at 8:00 AM)"
        NEXT_RUN=$(gcloud scheduler jobs describe scene-generation-daily \
            --location=$REGION \
            --format='value(scheduleTime)' 2>/dev/null || echo "unknown")
        echo "  Next run: $NEXT_RUN"
    else
        log_warn "Scheduler: PAUSED"
    fi
    echo ""

    # Check EventArc triggers
    echo "--- EventArc Triggers ---"

    # USD Assembly trigger (fires on .stage1_complete)
    USD_TRIGGER=$(gcloud eventarc triggers describe usd-assembly-trigger \
        --location=$REGION --format='value(name)' 2>/dev/null || echo "")
    if [ -n "$USD_TRIGGER" ]; then
        log_info "USD Assembly trigger: ACTIVE"
    else
        log_warn "USD Assembly trigger: NOT FOUND"
    fi

    # Episode Generation trigger (fires on .usd_complete)
    EPISODE_TRIGGER=$(gcloud eventarc triggers describe episode-generation-trigger \
        --location=$REGION --format='value(name)' 2>/dev/null || echo "")
    if [ -n "$EPISODE_TRIGGER" ]; then
        log_info "Episode Generation trigger: ACTIVE"
    else
        log_warn "Episode Generation trigger: NOT FOUND"
    fi

    # Upsell trigger (should NOT exist - handled inline)
    UPSELL_TRIGGER=$(gcloud eventarc triggers describe upsell-features-trigger \
        --location=$REGION --format='value(name)' 2>/dev/null || echo "")
    if [ -n "$UPSELL_TRIGGER" ]; then
        log_warn "Upsell Features trigger: ACTIVE (SHOULD BE REMOVED - inline processing is used)"
        echo "  Remove with: gcloud eventarc triggers delete upsell-features-trigger --location=$REGION"
    else
        log_info "Upsell Features trigger: NOT PRESENT (correct - inline processing)"
    fi
    echo ""

    # Check workflows
    echo "--- Deployed Workflows ---"
    gcloud workflows list --location=$REGION --format='table(name,state,updateTime)' 2>/dev/null || log_warn "Could not list workflows"
    echo ""

    # Check Cloud Run jobs
    echo "--- Cloud Run Jobs ---"
    gcloud run jobs list --region=$REGION --format='table(name,status.conditions[0].status)' 2>/dev/null || log_warn "Could not list Cloud Run jobs"
    echo ""
}

# ============================================================================
# Enable Pipeline
# ============================================================================

enable_pipeline() {
    echo "=============================================="
    echo "Enabling BlueprintPipeline Automation"
    echo "=============================================="
    echo ""

    # Enable scene generation scheduler
    log_info "Enabling scene generation scheduler..."
    if gcloud scheduler jobs resume scene-generation-daily --location=$REGION 2>/dev/null; then
        log_info "Scheduler enabled successfully"
    else
        log_warn "Scheduler not found. Run 'setup' first."
    fi
    echo ""

    # Verify EventArc triggers are in place
    log_info "Checking EventArc triggers..."

    if ! gcloud eventarc triggers describe episode-generation-trigger --location=$REGION &>/dev/null; then
        log_warn "Episode generation trigger missing. Creating..."
        create_episode_generation_trigger
    else
        log_info "Episode generation trigger is active"
    fi

    # Remove upsell trigger if it exists (inline processing is preferred)
    if gcloud eventarc triggers describe upsell-features-trigger --location=$REGION &>/dev/null; then
        log_warn "Removing upsell-features-trigger (inline processing is used)..."
        gcloud eventarc triggers delete upsell-features-trigger --location=$REGION --quiet
        log_info "Upsell trigger removed"
    fi

    echo ""
    log_info "Pipeline automation enabled!"
    echo ""
    echo "The pipeline will now automatically:"
    echo "  1. Generate scenes daily at 8:00 AM (Cloud Scheduler)"
    echo "  2. Process 3D reconstruction (external)"
    echo "  3. Assemble USD files (on .stage1_complete)"
    echo "  4. Generate episodes with upsell features (on .usd_complete)"
    echo ""
}

# ============================================================================
# Disable Pipeline
# ============================================================================

disable_pipeline() {
    echo "=============================================="
    echo "Disabling BlueprintPipeline Automation"
    echo "=============================================="
    echo ""

    # Pause scene generation scheduler
    log_info "Pausing scene generation scheduler..."
    if gcloud scheduler jobs pause scene-generation-daily --location=$REGION 2>/dev/null; then
        log_info "Scheduler paused successfully"
    else
        log_warn "Scheduler not found or already paused"
    fi

    echo ""
    log_info "Pipeline automation disabled"
    echo ""
    echo "Note: EventArc triggers remain active but won't fire without"
    echo "      new completion markers from the scene generation job."
    echo ""
    echo "To completely remove triggers, delete them manually:"
    echo "  gcloud eventarc triggers delete episode-generation-trigger --location=$REGION"
    echo ""
}

# ============================================================================
# Setup All Triggers
# ============================================================================

setup_all() {
    echo "=============================================="
    echo "Setting Up BlueprintPipeline Triggers"
    echo "=============================================="
    echo ""

    # Create service accounts if needed
    log_info "Checking service accounts..."
    if ! gcloud iam service-accounts describe ${WORKFLOW_SA}@${PROJECT_ID}.iam.gserviceaccount.com &>/dev/null; then
        log_info "Creating workflow service account..."
        gcloud iam service-accounts create $WORKFLOW_SA \
            --display-name="Blueprint Pipeline Workflow SA"
    fi
    echo ""

    # Create scheduler
    log_info "Creating scene generation scheduler..."
    if gcloud scheduler jobs describe scene-generation-daily --location=$REGION &>/dev/null; then
        log_warn "Scheduler already exists"
    else
        gcloud scheduler jobs create http scene-generation-daily \
            --schedule="0 8 * * *" \
            --location=$REGION \
            --uri="https://workflowexecutions.googleapis.com/v1/projects/${PROJECT_ID}/locations/${REGION}/workflows/scene-generation-pipeline/executions" \
            --http-method=POST \
            --oauth-service-account-email="${SCHEDULER_SA}@${PROJECT_ID}.iam.gserviceaccount.com" \
            --message-body='{"argument": "{\"scenes_per_run\": 10}"}' \
            --paused
        log_info "Scheduler created (paused)"
    fi
    echo ""

    # Create EventArc triggers
    log_info "Creating EventArc triggers..."

    # Episode generation trigger
    create_episode_generation_trigger

    echo ""
    log_info "Setup complete!"
    echo ""
    echo "To enable automated pipeline, run:"
    echo "  ./scripts/enable_pipeline.sh enable"
    echo ""
}

create_episode_generation_trigger() {
    if gcloud eventarc triggers describe episode-generation-trigger --location=$REGION &>/dev/null; then
        log_warn "Episode generation trigger already exists"
        return
    fi

    gcloud eventarc triggers create episode-generation-trigger \
        --location=$REGION \
        --service-account="${WORKFLOW_SA}@${PROJECT_ID}.iam.gserviceaccount.com" \
        --destination-workflow=episode-generation-pipeline \
        --destination-workflow-location=$REGION \
        --event-filters="type=google.cloud.storage.object.v1.finalized" \
        --event-filters="bucket=${BUCKET}" \
        --event-data-content-type="application/json"

    log_info "Episode generation trigger created"
}

# ============================================================================
# Manual Trigger Helpers
# ============================================================================

trigger_scene() {
    SCENE_ID=$1
    TIER=${2:-"standard"}

    if [ -z "$SCENE_ID" ]; then
        log_error "Usage: $0 trigger-scene <scene_id> [tier]"
        exit 1
    fi

    log_info "Triggering episode generation for scene: $SCENE_ID (tier: $TIER)"

    gcloud workflows run episode-generation-pipeline \
        --location=$REGION \
        --data="{\"data\":{\"bucket\":\"${BUCKET}\",\"name\":\"scenes/${SCENE_ID}/usd/.usd_complete\"},\"bundle_tier\":\"${TIER}\"}"

    log_info "Workflow triggered. Monitor with:"
    echo "  gcloud workflows executions list episode-generation-pipeline --location=$REGION"
}

trigger_upsell() {
    SCENE_ID=$1
    TIER=${2:-"pro"}
    FORCE=${3:-"false"}

    if [ -z "$SCENE_ID" ]; then
        log_error "Usage: $0 trigger-upsell <scene_id> [tier] [force]"
        exit 1
    fi

    log_info "Triggering upsell processing for scene: $SCENE_ID (tier: $TIER, force: $FORCE)"

    gcloud workflows run upsell-features-pipeline \
        --location=$REGION \
        --data="{\"data\":{\"bucket\":\"${BUCKET}\",\"name\":\"scenes/${SCENE_ID}/episodes/.episodes_complete\"},\"bundle_tier\":\"${TIER}\",\"force_reprocess\":${FORCE}}"

    log_info "Workflow triggered. Monitor with:"
    echo "  gcloud workflows executions list upsell-features-pipeline --location=$REGION"
}

# ============================================================================
# Main
# ============================================================================

case "${1:-status}" in
    enable)
        enable_pipeline
        ;;
    disable)
        disable_pipeline
        ;;
    status)
        check_status
        ;;
    setup)
        setup_all
        ;;
    trigger-scene)
        trigger_scene "$2" "$3"
        ;;
    trigger-upsell)
        trigger_upsell "$2" "$3" "$4"
        ;;
    *)
        echo "Usage: $0 {enable|disable|status|setup|trigger-scene|trigger-upsell}"
        echo ""
        echo "Commands:"
        echo "  enable          Enable automated pipeline (resume scheduler)"
        echo "  disable         Disable automated pipeline (pause scheduler)"
        echo "  status          Check status of all pipeline components"
        echo "  setup           Initial setup of all triggers and scheduler"
        echo "  trigger-scene   Manually trigger episode generation"
        echo "                  Usage: trigger-scene <scene_id> [tier]"
        echo "  trigger-upsell  Manually trigger upsell processing"
        echo "                  Usage: trigger-upsell <scene_id> [tier] [force]"
        echo ""
        exit 1
        ;;
esac
