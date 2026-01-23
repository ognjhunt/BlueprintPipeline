# =============================================================================
# Retention cleanup workflow + scheduler
# =============================================================================

data "google_project" "current" {
  project_id = var.project_id
}

locals {
  retention_cleanup_bucket = var.retention_cleanup_bucket != "" ? var.retention_cleanup_bucket : google_storage_bucket.pipeline_data.name
}

resource "google_workflows_workflow" "retention_cleanup" {
  name            = "retention-cleanup"
  project         = var.project_id
  region          = var.region
  description     = "Pipeline retention cleanup"
  service_account = google_service_account.workflow.email
  source_contents = file("${path.module}/../../workflows/retention-cleanup.yaml")

  depends_on = [google_project_service.apis]
}

resource "google_service_account_iam_member" "scheduler_token_creator" {
  service_account_id = google_service_account.workflow.name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = "serviceAccount:service-${data.google_project.current.number}@gcp-sa-cloudscheduler.iam.gserviceaccount.com"
}

resource "google_cloud_scheduler_job" "retention_cleanup_daily" {
  name        = "retention-cleanup-daily"
  project     = var.project_id
  region      = var.region
  description = "Daily retention cleanup workflow trigger"
  schedule    = var.retention_cleanup_schedule
  time_zone   = var.retention_cleanup_time_zone

  http_target {
    http_method = "POST"
    uri         = "https://workflowexecutions.googleapis.com/v1/projects/${var.project_id}/locations/${var.region}/workflows/${google_workflows_workflow.retention_cleanup.name}/executions"

    headers = {
      "Content-Type" = "application/json"
    }

    body = base64encode(jsonencode({
      argument = jsonencode({
        bucket = local.retention_cleanup_bucket
      })
    }))

    oauth_token {
      service_account_email = google_service_account.workflow.email
      scope                 = "https://www.googleapis.com/auth/cloud-platform"
    }
  }

  depends_on = [
    google_project_service.apis,
    google_workflows_workflow.retention_cleanup,
    google_service_account_iam_member.scheduler_token_creator,
  ]
}
