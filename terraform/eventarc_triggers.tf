# Eventarc trigger for hunyuan-pipeline
# Triggers when scene_assets.json is created
resource "google_eventarc_trigger" "hunyuan_trigger" {
  name     = "hunyuan-trigger"
  location = "us"  # Multi-regional for Cloud Storage

  matching_criteria {
    attribute = "type"
    value     = "google.cloud.storage.object.v1.finalized"
  }

  matching_criteria {
    attribute = "bucket"
    value     = var.bucket_name
  }

  destination {
    workflow = data.google_workflows_workflow.hunyuan_pipeline.id
  }

  service_account = var.service_account_email

  labels = {
    managed_by = "terraform"
    pipeline   = "hunyuan"
  }
}

# Eventarc trigger for sam3d-pipeline
# Triggers when scene_assets.json is created
resource "google_eventarc_trigger" "sam3d_trigger" {
  name     = "sam3d-trigger"
  location = "us"  # Multi-regional for Cloud Storage

  matching_criteria {
    attribute = "type"
    value     = "google.cloud.storage.object.v1.finalized"
  }

  matching_criteria {
    attribute = "bucket"
    value     = var.bucket_name
  }

  destination {
    workflow = data.google_workflows_workflow.sam3d_pipeline.id
  }

  service_account = var.service_account_email

  labels = {
    managed_by = "terraform"
    pipeline   = "sam3d"
  }
}

# Eventarc trigger for usd-assembly-pipeline
# Triggers when scene_assets.json is created
resource "google_eventarc_trigger" "usd_assembly_trigger" {
  name     = "usd-assembly-trigger"
  location = "us"  # Multi-regional for Cloud Storage

  matching_criteria {
    attribute = "type"
    value     = "google.cloud.storage.object.v1.finalized"
  }

  matching_criteria {
    attribute = "bucket"
    value     = var.bucket_name
  }

  destination {
    workflow = data.google_workflows_workflow.usd_assembly_pipeline.id
  }

  service_account = var.service_account_email

  labels = {
    managed_by = "terraform"
    pipeline   = "usd-assembly"
  }
}

# Note: interactive-trigger already exists, but including it here for completeness
# You can import the existing one or manage it separately
resource "google_eventarc_trigger" "interactive_trigger" {
  name     = "interactive-trigger"
  location = "us"  # Multi-regional for Cloud Storage

  matching_criteria {
    attribute = "type"
    value     = "google.cloud.storage.object.v1.finalized"
  }

  matching_criteria {
    attribute = "bucket"
    value     = var.bucket_name
  }

  destination {
    workflow = data.google_workflows_workflow.interactive_pipeline.id
  }

  service_account = var.service_account_email

  labels = {
    managed_by = "terraform"
    pipeline   = "interactive"
  }
}
