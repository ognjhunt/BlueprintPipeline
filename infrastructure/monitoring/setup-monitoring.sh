#!/bin/bash
# =============================================================================
# BlueprintPipeline - Monitoring Setup Script
# =============================================================================
# Sets up Cloud Monitoring dashboards and alert policies.
#
# Usage:
#   ./setup-monitoring.sh --project YOUR_PROJECT_ID [--email alerts@example.com]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Defaults
PROJECT_ID=""
NOTIFICATION_EMAIL=""
CLUSTER_NAME="blueprint-cluster"
DRY_RUN=false

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Sets up Cloud Monitoring for BlueprintPipeline.

Required:
  --project PROJECT_ID     GCP project ID

Options:
  --email EMAIL           Email for alert notifications
  --cluster NAME          GKE cluster name (default: blueprint-cluster)
  --dry-run              Show what would be done without executing
  --help                  Show this help message

Examples:
  $(basename "$0") --project my-project --email alerts@example.com
  $(basename "$0") --project my-project --dry-run

EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --project)
            PROJECT_ID="$2"
            shift 2
            ;;
        --email)
            NOTIFICATION_EMAIL="$2"
            shift 2
            ;;
        --cluster)
            CLUSTER_NAME="$2"
            shift 2
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

if [[ -z "$PROJECT_ID" ]]; then
    log_error "Project ID is required"
    usage
fi

# =============================================================================
# Setup Notification Channel
# =============================================================================

setup_notification_channel() {
    log_info "Setting up notification channel..."

    if [[ -z "$NOTIFICATION_EMAIL" ]]; then
        log_warning "No email provided, skipping notification channel"
        return 0
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would create notification channel for $NOTIFICATION_EMAIL"
        return 0
    fi

    # Check if channel already exists
    EXISTING=$(gcloud alpha monitoring channels list \
        --project="$PROJECT_ID" \
        --filter="type=email AND labels.email_address=$NOTIFICATION_EMAIL" \
        --format="value(name)" 2>/dev/null || echo "")

    if [[ -n "$EXISTING" ]]; then
        log_info "Notification channel already exists: $EXISTING"
        NOTIFICATION_CHANNEL=$(echo "$EXISTING" | head -1 | rev | cut -d'/' -f1 | rev)
    else
        # Create new channel
        CHANNEL_RESULT=$(gcloud alpha monitoring channels create \
            --project="$PROJECT_ID" \
            --display-name="Blueprint Pipeline Alerts" \
            --type="email" \
            --channel-labels="email_address=$NOTIFICATION_EMAIL" \
            --format="value(name)")

        NOTIFICATION_CHANNEL=$(echo "$CHANNEL_RESULT" | rev | cut -d'/' -f1 | rev)
        log_success "Created notification channel: $NOTIFICATION_CHANNEL"
    fi

    export NOTIFICATION_CHANNEL
}

# =============================================================================
# Deploy Dashboards
# =============================================================================

deploy_dashboards() {
    log_info "Deploying Cloud Monitoring dashboards..."

    DASHBOARDS_DIR="${SCRIPT_DIR}/dashboards"

    for dashboard_file in "$DASHBOARDS_DIR"/*.json; do
        if [[ ! -f "$dashboard_file" ]]; then
            continue
        fi

        dashboard_name=$(basename "$dashboard_file" .json)
        log_info "Deploying dashboard: $dashboard_name"

        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "[DRY RUN] Would deploy $dashboard_file"
            continue
        fi

        # Check if dashboard exists
        EXISTING=$(gcloud monitoring dashboards list \
            --project="$PROJECT_ID" \
            --filter="displayName~'$dashboard_name'" \
            --format="value(name)" 2>/dev/null || echo "")

        if [[ -n "$EXISTING" ]]; then
            log_info "Updating existing dashboard..."
            DASHBOARD_ID=$(echo "$EXISTING" | head -1 | rev | cut -d'/' -f1 | rev)

            gcloud monitoring dashboards update "$DASHBOARD_ID" \
                --project="$PROJECT_ID" \
                --config-from-file="$dashboard_file" || {
                log_warning "Failed to update dashboard, trying create..."
                gcloud monitoring dashboards delete "$DASHBOARD_ID" \
                    --project="$PROJECT_ID" --quiet 2>/dev/null || true
                gcloud monitoring dashboards create \
                    --project="$PROJECT_ID" \
                    --config-from-file="$dashboard_file"
            }
        else
            gcloud monitoring dashboards create \
                --project="$PROJECT_ID" \
                --config-from-file="$dashboard_file"
        fi

        log_success "Deployed dashboard: $dashboard_name"
    done
}

# =============================================================================
# Deploy Alert Policies
# =============================================================================

deploy_alert_policies() {
    log_info "Deploying alert policies..."

    ALERTS_FILE="${SCRIPT_DIR}/alerts/alert-policies.yaml"

    if [[ ! -f "$ALERTS_FILE" ]]; then
        log_warning "Alert policies file not found: $ALERTS_FILE"
        return 0
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would deploy alert policies from $ALERTS_FILE"
        return 0
    fi

    # Substitute variables in the alerts file
    TEMP_FILE=$(mktemp)
    sed -e "s/\${PROJECT_ID}/$PROJECT_ID/g" \
        -e "s/\${NOTIFICATION_CHANNEL}/${NOTIFICATION_CHANNEL:-default}/g" \
        -e "s/\${BUCKET}/$PROJECT_ID-blueprint-data/g" \
        "$ALERTS_FILE" > "$TEMP_FILE"

    # Split YAML into individual policies and deploy each
    csplit -f "${TEMP_FILE}_" -z "$TEMP_FILE" '/^---$/' '{*}' 2>/dev/null || true

    for policy_file in ${TEMP_FILE}_*; do
        if [[ ! -f "$policy_file" ]] || [[ ! -s "$policy_file" ]]; then
            continue
        fi

        # Extract policy name
        policy_name=$(grep -m1 "displayName:" "$policy_file" | sed 's/.*displayName: *"\?\([^"]*\)"\?/\1/' || echo "unknown")

        log_info "Deploying alert policy: $policy_name"

        # Convert YAML to JSON for gcloud
        if command -v yq &> /dev/null; then
            JSON_FILE="${policy_file}.json"
            yq -o json "$policy_file" > "$JSON_FILE" 2>/dev/null || continue

            # Create alert policy
            gcloud alpha monitoring policies create \
                --project="$PROJECT_ID" \
                --policy-from-file="$JSON_FILE" 2>/dev/null || {
                log_warning "Failed to create policy: $policy_name (may already exist)"
            }

            rm -f "$JSON_FILE"
        else
            log_warning "yq not installed, skipping YAML->JSON conversion"
            log_info "Install with: brew install yq"
        fi

        rm -f "$policy_file"
    done

    rm -f "$TEMP_FILE"
    log_success "Alert policies deployed"
}

# =============================================================================
# Create Custom Metrics
# =============================================================================

create_custom_metrics() {
    log_info "Creating custom metrics descriptors..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would create custom metrics"
        return 0
    fi

    # Pipeline job duration metric
    gcloud monitoring metrics-descriptors create \
        "custom.googleapis.com/blueprint/job_duration_seconds" \
        --project="$PROJECT_ID" \
        --display-name="Pipeline Job Duration" \
        --description="Duration of pipeline job execution in seconds" \
        --metric-kind="gauge" \
        --value-type="double" \
        --labels="job_type:STRING,scene_id:STRING,status:STRING" \
        2>/dev/null || log_info "Metric already exists: job_duration_seconds"

    # Episode generation count metric
    gcloud monitoring metrics-descriptors create \
        "custom.googleapis.com/blueprint/episodes_generated" \
        --project="$PROJECT_ID" \
        --display-name="Episodes Generated" \
        --description="Number of episodes generated" \
        --metric-kind="cumulative" \
        --value-type="int64" \
        --labels="scene_id:STRING,robot_type:STRING,quality:STRING" \
        2>/dev/null || log_info "Metric already exists: episodes_generated"

    # Quality score metric
    gcloud monitoring metrics-descriptors create \
        "custom.googleapis.com/blueprint/episode_quality_score" \
        --project="$PROJECT_ID" \
        --display-name="Episode Quality Score" \
        --description="Quality score of generated episodes" \
        --metric-kind="gauge" \
        --value-type="double" \
        --labels="scene_id:STRING,task_type:STRING" \
        2>/dev/null || log_info "Metric already exists: episode_quality_score"

    log_success "Custom metrics created"
}

# =============================================================================
# Setup Logs-based Metrics
# =============================================================================

setup_log_metrics() {
    log_info "Setting up logs-based metrics..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would create logs-based metrics"
        return 0
    fi

    METRICS_DIR="${SCRIPT_DIR}/metrics"

    if [[ -d "$METRICS_DIR" ]]; then
        for metric_file in "$METRICS_DIR"/*.yaml; do
            if [[ ! -f "$metric_file" ]]; then
                continue
            fi

            metric_name=$(grep -m1 "^name:" "$metric_file" | awk '{print $2}' || echo "$(basename "$metric_file")")
            log_info "Creating logs-based metric from config: $metric_name"

            gcloud logging metrics create \
                --project="$PROJECT_ID" \
                --config-from-file="$metric_file" \
                2>/dev/null || log_info "Metric already exists: $metric_name"
        done
    else
        log_warning "Metrics config directory not found: $METRICS_DIR"
    fi

    # Error count by job type
    gcloud logging metrics create blueprint_error_count \
        --project="$PROJECT_ID" \
        --description="Count of errors in BlueprintPipeline" \
        --log-filter='resource.type="k8s_container"
            resource.labels.namespace_name="blueprint"
            severity>=ERROR' \
        2>/dev/null || log_info "Metric already exists: blueprint_error_count"

    # Job completion latency
    gcloud logging metrics create blueprint_job_latency \
        --project="$PROJECT_ID" \
        --description="Pipeline job completion latency" \
        --log-filter='resource.type="k8s_container"
            resource.labels.namespace_name="blueprint"
            textPayload=~"completed in"' \
        2>/dev/null || log_info "Metric already exists: blueprint_job_latency"

    log_success "Logs-based metrics created"
}

# =============================================================================
# Main
# =============================================================================

main() {
    echo ""
    echo "============================================================"
    echo "BlueprintPipeline Monitoring Setup"
    echo "============================================================"
    echo "Project: $PROJECT_ID"
    echo "Email: ${NOTIFICATION_EMAIL:-[not set]}"
    echo "Dry Run: $DRY_RUN"
    echo "============================================================"
    echo ""

    setup_notification_channel
    deploy_dashboards
    deploy_alert_policies
    create_custom_metrics
    setup_log_metrics

    echo ""
    echo "============================================================"
    echo -e "${GREEN}Monitoring setup complete!${NC}"
    echo "============================================================"
    echo ""
    echo "View dashboards:"
    echo "  https://console.cloud.google.com/monitoring/dashboards?project=$PROJECT_ID"
    echo ""
    echo "View alerts:"
    echo "  https://console.cloud.google.com/monitoring/alerting?project=$PROJECT_ID"
    echo ""
}

main
