# =============================================================================
# Secret rotation automation (Cloud Scheduler + Cloud Run Job)
# =============================================================================

resource "google_service_account" "secret_rotation_job" {
  account_id   = "secret-rotation-job"
  display_name = "Secret Rotation Job Service Account"
  project      = var.project_id

  depends_on = [google_project_service.apis]
}

resource "google_service_account" "secret_rotation_scheduler" {
  account_id   = "secret-rotation-scheduler"
  display_name = "Secret Rotation Scheduler Service Account"
  project      = var.project_id

  depends_on = [google_project_service.apis]
}

resource "google_project_iam_member" "secret_rotation_job_roles" {
  for_each = toset([
    "roles/secretmanager.secretVersionAdder",
    "roles/secretmanager.secretAccessor",
    "roles/logging.logWriter",
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.secret_rotation_job.email}"

  depends_on = [google_project_service.apis]
}

resource "google_cloud_run_v2_job" "secret_rotation" {
  name     = "secret-rotation-job"
  location = var.region
  project  = var.project_id

  template {
    template {
      service_account = google_service_account.secret_rotation_job.email

      containers {
        image = var.secret_rotation_job_image

        env {
          name  = "ROTATION_SECRET_IDS"
          value = join(",", var.secret_rotation_secret_ids)
        }

        env {
          name  = "ROTATION_BYTE_LENGTH"
          value = tostring(var.secret_rotation_byte_length)
        }

        env {
          name  = "ROTATION_REASON"
          value = "scheduled"
        }

        env {
          name  = "ROTATION_ACTOR"
          value = "cloud-scheduler"
        }

        env {
          name  = "GCP_PROJECT"
          value = var.project_id
        }
      }
    }
  }

  depends_on = [google_project_service.apis]
}

resource "google_cloud_run_v2_job_iam_member" "secret_rotation_invoker" {
  name     = google_cloud_run_v2_job.secret_rotation.name
  location = var.region
  project  = var.project_id
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.secret_rotation_scheduler.email}"

  depends_on = [google_project_service.apis]
}

resource "google_cloud_scheduler_job" "secret_rotation" {
  name        = "secret-rotation-schedule"
  description = "Rotate Secret Manager secrets on a schedule"
  schedule    = var.secret_rotation_schedule
  time_zone   = var.secret_rotation_time_zone
  region      = var.region

  http_target {
    http_method = "POST"
    uri         = "https://run.googleapis.com/v2/projects/${var.project_id}/locations/${var.region}/jobs/${google_cloud_run_v2_job.secret_rotation.name}:run"

    oidc_token {
      service_account_email = google_service_account.secret_rotation_scheduler.email
      audience              = "https://run.googleapis.com/"
    }
  }

  depends_on = [google_cloud_run_v2_job_iam_member.secret_rotation_invoker]
}
