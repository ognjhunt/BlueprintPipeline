# =============================================================================
# Firebase Storage lifecycle (optional)
# =============================================================================

resource "google_storage_bucket" "firebase_storage" {
  count    = var.firebase_storage_bucket != "" ? 1 : 0
  name     = var.firebase_storage_bucket
  project  = var.project_id
  location = var.firebase_storage_bucket_location

  force_destroy = false

  lifecycle_rule {
    condition {
      age            = 365
      matches_prefix = ["datasets/"]
    }
    action {
      type = "Delete"
    }
  }

  lifecycle_rule {
    condition {
      age            = 180
      matches_prefix = ["logs/", "log/"]
    }
    action {
      type = "Delete"
    }
  }

  labels = {
    environment = var.environment
    managed_by  = "terraform"
    purpose     = "firebase-storage"
  }

  depends_on = [google_project_service.apis]
}
