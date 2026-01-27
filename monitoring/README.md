# BlueprintPipeline Monitoring Dashboard

This directory contains monitoring and alerting configuration for the BlueprintPipeline.

## Dashboard Setup

### Automated setup (recommended)

Run the deployment script to create dashboards, alert policies, and log-based metrics in one step:

```bash
./infrastructure/monitoring/setup-monitoring.sh --project YOUR_PROJECT_ID --email alerts@example.com
```

### Google Cloud Monitoring

1. **Import Dashboard**:
   ```bash
   gcloud monitoring dashboards create --config-from-file=dashboard_config.json
   ```

2. **View Dashboard**:
   - Go to: https://console.cloud.google.com/monitoring/dashboards
   - Select "BlueprintPipeline - Genie Sim 3.0 Dashboard"

## Monitoring gate requirements

The episode-generation workflow enforces a monitoring gate in production that checks for specific Cloud Monitoring
resources by name. Non-production environments can bypass failures by setting `MONITORING_GATE_STRICT=false`.

**Dashboards (exact names)**
- `BlueprintPipeline - Overview`
- `BlueprintPipeline - GPU Metrics`

**Alert policies (exact names)**
- `[Blueprint] Workflow Job Timeout Detected`
- `[Blueprint] Workflow Job Retry Spike`

**Log-based metrics (exact names)**
- `blueprint_job_timeout_events`
- `blueprint_job_retry_exhausted_total`
- `blueprint_job_timeout_usage_ratio`
- `blueprint_geniesim_sla_violations`
- `blueprint_job_failure_events`

### Prometheus + Grafana

1. **Start Prometheus**:
   ```bash
   # Add to prometheus.yml:
   scrape_configs:
     - job_name: 'blueprint_pipeline'
       static_configs:
         - targets: ['localhost:8000']

   # Start Prometheus
   prometheus --config.file=prometheus.yml
   ```

2. **Import Grafana Dashboard**:
   ```bash
   # In Grafana: Dashboards -> New -> Import
   # Upload monitoring/grafana_dashboard.json or paste its JSON content.
   # Select your Prometheus datasource when prompted.
   ```

### Grafana Panels (Prometheus)

The Grafana dashboard includes the following Genie Sim quality/performance panels:

- **Collision-Free Rate** (requires `collision_free_rate`)
- **Task Success Rate** (requires `task_success_rate`)
- **Genie Sim Episodes per Hour** (rate of `geniesim_episodes_generated_total`)

## Key Metrics

### Pipeline Performance
- **pipeline_runs_total**: Total pipeline runs (by job, status)
  - Target: 99% success rate
- **pipeline_duration_seconds**: Pipeline job duration
  - Monitor: P95 should be < 10 minutes per scene
- **scenes_in_progress**: Concurrent scenes being processed
  - Limit: Based on max_concurrent setting
- **blueprint_job_invocation_total**: Workflow job invocations emitted via `bp_metric`
  - SLO: 99% end-to-end success rate across workflows
- **blueprint_job_invocation_duration_seconds**: Stage duration from `bp_metric` logs
  - SLO: P95 < 900s per stage

### API Usage
- **gemini_api_calls_total**: Total Gemini API calls
- **gemini_tokens_input_total**: Total input tokens
- **gemini_tokens_output_total**: Total output tokens
- **gemini_api_latency_seconds**: Gemini API latency
  - Alert: If P95 > 5 seconds

### Genie Sim Integration
- **geniesim_jobs_submitted_total**: Jobs submitted to Genie Sim
- **geniesim_episodes_generated_total**: Episodes generated
- **geniesim_job_duration_seconds**: Job duration
- **geniesim_server_latency_seconds**: Server latency
  - Alert: If unavailable for > 5 minutes
- **geniesim_episodes_generated_total** (rate): Episodes/hour panel in Grafana

**Job.json metrics export**
- The `geniesim/job.json` summaries (`job_metrics_by_robot`, `job_metrics`, or `job_metrics_summary`) are exported as:
  - `geniesim_jobs_submitted_total` (counter, 1 per summary)
  - `geniesim_episodes_generated_total` (counter, uses `episodes_collected` or `total_episodes`)
  - `geniesim_job_duration_seconds` (histogram, uses `duration_seconds`)
- Labels are emitted consistently: `scene_id`, `job_id`, `robot_type`.
- Validate in Cloud Monitoring by filtering `custom.googleapis.com/blueprint_pipeline/geniesim_*` and checking labels
  for the scene/job you just ran (Metrics Explorer or `gcloud monitoring time-series list` with a label filter).

### Quality Metrics
- **episode_quality_score**: Episode quality distribution
  - Alert: If average < 0.7
- **collision_free_rate**: Collision-free episode rate
- **task_success_rate**: Task success rate
- **physics_validation_score**: Physics validation scores
- **blueprint_quality_gate_failures_total**: Quality gate failure count
  - SLO: Quality pass rate >= 98%

### Resource Usage
- **objects_processed_total**: Objects processed
- **storage_bytes_written_total**: Storage written
- **pipeline_cost_total_usd**: Total pipeline cost emitted by cost tracking
- **errors_total**: Errors by type
- **retries_total**: Retry attempts

### Delivery Integrity Audit
- **pipeline_runs_total** (`job=dataset-delivery-integrity-audit-job`): Per-bundle integrity pass/fail counts
- **errors_total** (`job=dataset-delivery-integrity-audit-job`): Audit failures by error type
- **storage_bytes_written_total** (`job=dataset-delivery-integrity-audit-job`): Bytes written for audit reports

### Cost Metrics
- **cost_per_scene**: Rolling total cost per scene (USD)

## Alerts

### Application Alerting (Webhook)

BlueprintPipeline includes a lightweight alerting helper that can post JSON payloads to a webhook
endpoint when health checks fail or jobs crash.

**Environment variables**
- `ALERT_BACKEND`: Alert backend selector (`webhook` or `none`, default: `none`).
- `ALERT_WEBHOOK_URL`: Destination webhook URL for alert payloads (required when `ALERT_BACKEND=webhook`).
- `ALERT_WEBHOOK_TIMEOUT_SECONDS`: Webhook timeout in seconds (positive number; default: `10`).
- `ALERT_MIN_SEVERITY`: Minimum severity to emit (`info`, `warning`, `error`, `critical`; default: `warning`).
- `ALERT_SOURCE`: Identifier included in the payload (default: `blueprint_pipeline`).
- `ALERT_HEALTHCHECK_ENABLED`: Enable health check alerts (`true`/`false`, default: `true`).
- `ALERT_HEALTHCHECK_FAILURE_THRESHOLD`: Minimum failing dependency count before alerting (default: `1`).
- `ALERT_HEALTHCHECK_SEVERITY`: Severity for health check alerts (default: `critical`).
- `ALERT_JOB_EXCEPTION_SEVERITY`: Severity for fatal job exception alerts (default: `critical`).

**Sample webhook payload**
```json
{
  "event_type": "particulate_healthcheck_dependency_check_failed",
  "summary": "Particulate health check failed",
  "details": {
    "service": "particulate",
    "failure_count": 1,
    "details": {
      "errors": []
    }
  },
  "severity": "critical",
  "source": "blueprint_pipeline",
  "timestamp": "2026-01-01T00:00:00Z"
}
```

### Critical Alerts

1. **Pipeline Failure Rate > 5%**:
   ```yaml
   condition:
     - metric: pipeline_runs_total{status="failure"}
     - threshold: rate > 0.05
   notification: PagerDuty
   ```

2. **Genie Sim Server Unavailable**:
   ```yaml
   condition:
     - metric: geniesim_server_latency_seconds
     - threshold: no data for 5 minutes
   notification: Slack
   ```

3. **Episode Quality Below Threshold**:
   ```yaml
   condition:
     - metric: episode_quality_score
     - threshold: average < 0.6
   notification: Email
   ```

4. **Pipeline Failure Rate (MQL)**:
   ```yaml
   condition:
     - metric: pipeline_runs_total
     - threshold: failure_rate > 0.05 over 30m
   notification: PagerDuty
   ```

### Warning Alerts

1. **High API Latency**:
   ```yaml
   condition:
     - metric: gemini_api_latency_seconds
     - threshold: p95 > 5s for 10 minutes
   notification: Slack
   ```

2. **Low Diversity Score**:
   ```yaml
   condition:
     - metric: episode_diversity_score
     - threshold: < 0.4 for any scene
   notification: Email
   ```

### Cost Alerts

1. **Cost Anomaly (Hourly Spike)**:
   ```yaml
   condition:
     - metric: pipeline_cost_total_usd
     - threshold: hourly_cost > 1.5x rolling 24h avg
   notification: Slack
   ```

## Setting Up Alerts

## Delivery Integrity Audit (Weekly)

The delivery integrity audit verifies delivered bundles against their `checksums.json` manifests and writes a report to
GCS. The workflow is defined in `workflows/delivery-integrity-audit.yaml` and is intended to be invoked weekly via
Cloud Scheduler (for example: `0 3 * * 1` for Mondays at 03:00 UTC).【F:workflows/delivery-integrity-audit.yaml†L1-L170】

### Expected Outputs
- **Audit report**: `gs://<audit-report-bucket>/audit-reports/delivery-integrity/<timestamp>_integrity_audit.json`
  - Includes per-bundle pass/fail status, missing files, checksum mismatches, and size mismatches.
- **Logs**: Structured audit events with `bp_metric=delivery_integrity_audit`, plus workflow start/complete entries.
- **Metrics**: `pipeline_runs_total`, `errors_total`, and `storage_bytes_written_total` labeled with the audit job.

### Remediation Steps
1. **Identify failing bundles** in the audit report under `bundles` or `errors`.
2. **Re-run dataset delivery** for the affected scene/job to regenerate the bundle in the delivery bucket.
3. **Recompute checksums** if source files changed, then re-run the audit.
4. **Escalate storage corruption** if checksum mismatches persist after re-delivery.

### Google Cloud Monitoring

#### Log-based Metrics (Workflow Timeouts/Retry Spikes)

```bash
# Create log-based metrics for timeout usage and retry exhaustion
gcloud logging metrics create --config-from-file=../infrastructure/monitoring/metrics/job-timeout-events.yaml
gcloud logging metrics create --config-from-file=../infrastructure/monitoring/metrics/job-retry-exhausted.yaml
gcloud logging metrics create --config-from-file=../infrastructure/monitoring/metrics/job-timeout-usage.yaml
gcloud logging metrics create --config-from-file=../infrastructure/monitoring/metrics/job-invocation-total.yaml
gcloud logging metrics create --config-from-file=../infrastructure/monitoring/metrics/job-invocation-duration.yaml
gcloud logging metrics create --config-from-file=../infrastructure/monitoring/metrics/quality-gate-failures.yaml
```

```bash
# Create alert policy
gcloud alpha monitoring policies create \
  --notification-channels=CHANNEL_ID \
  --display-name="Pipeline Failure Rate" \
  --condition-display-name="Failure rate > 5%" \
  --condition-expression='
    resource.type = "global"
    AND metric.type = "custom.googleapis.com/blueprint_pipeline/pipeline_runs_total"
    AND metric.label.status = "failure"
  ' \
  --condition-threshold-value=0.05 \
  --condition-threshold-duration=300s
```

### Prometheus Alertmanager

Create `alerts.yml`:
```yaml
groups:
  - name: blueprint_pipeline
    interval: 30s
    rules:
      - alert: HighPipelineFailureRate
        expr: rate(pipeline_runs_total{status="failure"}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High pipeline failure rate"
          description: "Pipeline failure rate is {{ $value }}%"

      - alert: GenieSimServerUnavailable
        expr: up{job="blueprint_pipeline"} == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Genie Sim server unavailable"
```

## Cost Monitoring

In addition to operational metrics, track costs using the cost tracking module:

```python
from tools.cost_tracking import get_cost_tracker

tracker = get_cost_tracker()

# Get period costs
costs = tracker.get_period_cost(days=7)
print(f"Total cost (7 days): ${costs['total']:.2f}")
print(f"Avg per scene: ${costs['avg_per_scene']:.2f}")

# Get scene cost breakdown
breakdown = tracker.get_scene_cost("scene_001")
print(f"Gemini: ${breakdown.gemini:.4f}")
print(f"Genie Sim: ${breakdown.geniesim:.4f}")
print(f"Compute: ${breakdown.cloud_run:.4f}")
```

### Grafana dashboard JSON

The Prometheus/Grafana dashboard definition lives in `monitoring/grafana_dashboard.json`.
Import it in Grafana via **Dashboards → New → Import** and upload the JSON file, then
select your Prometheus datasource when prompted.

### Cost tracking configuration

Cost tracking reads canonical defaults from `tools/cost_tracking/pricing_defaults.json`
and then applies overrides in the following order: custom pricing JSON → environment
variable overrides.

* `GENIESIM_JOB_COST` and `GENIESIM_EPISODE_COST` must be set to real values in
  production (`BP_ENV` or `GENIESIM_ENV` set to `production`/`prod`), otherwise
  the tracker raises an error.
* `GENIESIM_GPU_RATE_TABLE` can contain JSON for per-region or per-node hourly
  GPU rates (for example, `{"default": {"g5.xlarge": 1.006}}`).
* `GENIESIM_GPU_RATE_TABLE_PATH` points at a JSON file with the same structure.
* `GENIESIM_GPU_HOURLY_RATE` provides a single fallback hourly rate if no table
  entry matches.
* `GENIESIM_GPU_REGION` and `GENIESIM_GPU_NODE_TYPE` can be used to select the
  hourly rate when job metadata does not include region or node type.
* `COST_TRACKING_PRICING_JSON` or `COST_TRACKING_PRICING_PATH` can override any
  pricing keys in the defaults file (for example, Cloud Run pricing).

## SLO Targets & On-Call Escalation

### SLO Targets
- **End-to-end success rate:** ≥ 99% over a rolling 1-hour window (Cloud Workflow executions). This is enforced by the alert policy `blueprint-slo-e2e-success-rate`.
- **Stage latency:** P95 job invocation duration ≤ 900s over a rolling 15-minute window (from `bp_metric` job_invocation duration_seconds). This is enforced by `blueprint-slo-stage-latency`.
- **Quality pass-rate:** ≥ 98% successful quality gate evaluations. Quality gates validate `validation_report.json` thresholds (pass_rate/average_score) and `quality_gate_report.json` can_proceed outcomes. This is enforced by `blueprint-slo-quality-pass-rate`.

### On-Call Escalation
1. **Initial page:** Primary on-call engineer via PagerDuty/Email for any SLO breach.
2. **15 minutes:** If user impact is confirmed or the SLO remains breached, escalate to the incident commander and post in `#blueprint-oncall`.
3. **30 minutes:** Engage service owners for the affected workflow(s) and start mitigation (rollback, scale, or disable downstream consumers).
4. **Post-incident:** File a follow-up with root-cause analysis and error budget impact within 24 hours.

## Dashboard Queries

### Success Rate
```
sum(rate(pipeline_runs_total{status="success"}[1h])) /
sum(rate(pipeline_runs_total[1h]))
```

### Average Processing Time
```
avg(rate(pipeline_duration_seconds_sum[1h]) /
    rate(pipeline_duration_seconds_count[1h]))
by (job)
```

### Episode Quality Average
```
avg(episode_quality_score)
```

### Cost per Scene (requires cost tracking data)
```python
# Use Python for cost analysis
costs = tracker.get_period_cost(days=1)
avg_cost = costs['avg_per_scene']
```

## Troubleshooting

### High Latency
1. Check Gemini API quota and rate limits
2. Verify network connectivity to Genie Sim
3. Review concurrent scene limit

### Low Quality Scores
1. Check episode diversity metrics
2. Review physics validation errors
3. Verify Isaac Sim is being used (not mock data)

### High Costs
1. Review cost breakdown by component
2. Check for excessive Gemini calls
3. Optimize physics estimation caching

## Next Steps

1. Set up notification channels (Slack, email, PagerDuty)
2. Configure alert policies based on SLOs
3. Create custom views for different team members
4. Set up automated reports (daily/weekly)
