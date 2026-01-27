# =============================================================================
# BlueprintPipeline - Monitoring (Dashboards, Alerts, Log Metrics)
# =============================================================================

locals {
  monitoring_dashboards = {
    overview = merge(
      jsondecode(file("${path.module}/../monitoring/dashboards/pipeline-overview.json")),
      { displayName = var.monitoring_dashboard_overview_name }
    )
    gpu_metrics = merge(
      jsondecode(file("${path.module}/../monitoring/dashboards/gpu-metrics.json")),
      { displayName = var.monitoring_dashboard_gpu_name }
    )
  }

  monitoring_metric_files = {
    job_timeout_events      = "${path.module}/../monitoring/metrics/job-timeout-events.yaml"
    job_retry_exhausted     = "${path.module}/../monitoring/metrics/job-retry-exhausted.yaml"
    job_timeout_usage_ratio = "${path.module}/../monitoring/metrics/job-timeout-usage.yaml"
    geniesim_sla_violations  = "${path.module}/../monitoring/metrics/geniesim-sla-violations.yaml"
    job_failure_events      = "${path.module}/../monitoring/metrics/job-failure-events.yaml"
  }

  monitoring_metrics = {
    for metric_name, metric_path in local.monitoring_metric_files :
    metric_name => yamldecode(file(metric_path))
  }
}

resource "google_monitoring_dashboard" "workflow_gate" {
  for_each = local.monitoring_dashboards

  project        = var.project_id
  dashboard_json = jsonencode(each.value)

  depends_on = [google_project_service.apis]
}

resource "google_logging_metric" "workflow_gate" {
  for_each = local.monitoring_metrics

  name        = each.value.name
  description = each.value.description
  filter      = trimspace(each.value.filter)

  metric_descriptor {
    metric_kind = each.value.metricDescriptor.metricKind
    value_type  = each.value.metricDescriptor.valueType
    unit        = each.value.metricDescriptor.unit
  }

  label_extractors = try(each.value.labelExtractors, null)
  value_extractor  = try(each.value.valueExtractor, null)

  depends_on = [google_project_service.apis]
}

resource "google_monitoring_alert_policy" "workflow_job_timeout" {
  display_name = var.monitoring_alert_workflow_timeout_name
  combiner     = "OR"

  documentation {
    content = <<-EOT
      A workflow job invocation hit or exceeded its configured timeout.

      Actions:
      1. Review workflow execution logs for duration and timeout details
      2. Confirm scene complexity or bundle tier inputs
      3. Adjust adaptive timeout overrides if needed
    EOT
    mime_type = "text/markdown"
  }

  conditions {
    display_name = "Job Timeout Events > 0"
    condition_threshold {
      filter          = "resource.type=\"global\" AND metric.type=\"logging.googleapis.com/user/blueprint_job_timeout_events\""
      comparison      = "COMPARISON_GT"
      threshold_value = 0
      duration        = "300s"

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_SUM"
        cross_series_reducer = "REDUCE_SUM"
      }
    }
  }

  notification_channels = var.monitoring_notification_channel_ids

  depends_on = [google_project_service.apis]
}

resource "google_monitoring_alert_policy" "workflow_job_retry_spike" {
  display_name = var.monitoring_alert_workflow_retry_spike_name
  combiner     = "OR"

  documentation {
    content = <<-EOT
      Workflow job invocations are exhausting retries at a high rate.

      Actions:
      1. Review upstream API health or Cloud Run errors
      2. Check recent deploys that could cause transient failures
      3. Consider adjusting retry/backoff or rate limiting
    EOT
    mime_type = "text/markdown"
  }

  conditions {
    display_name = "Retry Exhausted > 3 in 10 minutes"
    condition_threshold {
      filter          = "resource.type=\"global\" AND metric.type=\"logging.googleapis.com/user/blueprint_job_retry_exhausted_total\""
      comparison      = "COMPARISON_GT"
      threshold_value = 3
      duration        = "600s"

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_SUM"
        cross_series_reducer = "REDUCE_SUM"
      }
    }
  }

  notification_channels = var.monitoring_notification_channel_ids

  depends_on = [google_project_service.apis]
}
