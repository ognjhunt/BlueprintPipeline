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
   # Use grafana_dashboard.json (if available)
   # Or manually create dashboard using Prometheus metrics
   ```

## Key Metrics

### Pipeline Performance
- **pipeline_runs_total**: Total pipeline runs (by job, status)
  - Target: 99% success rate
- **pipeline_duration_seconds**: Pipeline job duration
  - Monitor: P95 should be < 10 minutes per scene
- **scenes_in_progress**: Concurrent scenes being processed
  - Limit: Based on max_concurrent setting

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
- **geniesim_api_latency_seconds**: API latency
  - Alert: If unavailable for > 5 minutes

### Quality Metrics
- **episode_quality_score**: Episode quality distribution
  - Alert: If average < 0.7
- **physics_validation_score**: Physics validation scores

### Resource Usage
- **objects_processed_total**: Objects processed
- **storage_bytes_written_total**: Storage written
- **errors_total**: Errors by type
- **retries_total**: Retry attempts

## Alerts

### Critical Alerts

1. **Pipeline Failure Rate > 5%**:
   ```yaml
   condition:
     - metric: pipeline_runs_total{status="failure"}
     - threshold: rate > 0.05
   notification: PagerDuty
   ```

2. **Genie Sim API Unavailable**:
   ```yaml
   condition:
     - metric: geniesim_api_latency_seconds
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

## Setting Up Alerts

### Google Cloud Monitoring

#### Log-based Metrics (Workflow Timeouts/Retry Spikes)

```bash
# Create log-based metrics for timeout usage and retry exhaustion
gcloud logging metrics create --config-from-file=../infrastructure/monitoring/metrics/job-timeout-events.yaml
gcloud logging metrics create --config-from-file=../infrastructure/monitoring/metrics/job-retry-exhausted.yaml
gcloud logging metrics create --config-from-file=../infrastructure/monitoring/metrics/job-timeout-usage.yaml
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

      - alert: GenieSimAPIUnavailable
        expr: up{job="blueprint_pipeline"} == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Genie Sim API unavailable"
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
