# =============================================================================
# BlueprintPipeline - GKE Infrastructure with GPU Support
# =============================================================================
# This Terraform configuration creates:
#   - GKE cluster with Autopilot disabled (for GPU support)
#   - GPU node pool with NVIDIA T4 GPUs
#   - VPC and networking
#   - Service accounts and IAM bindings
#   - Cloud Storage buckets
#   - Artifact Registry for container images
#
# Usage:
#   cd infrastructure/terraform
#   terraform init
#   terraform plan -var="project_id=YOUR_PROJECT"
#   terraform apply -var="project_id=YOUR_PROJECT"
# =============================================================================

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.25"
    }
  }

  # Remote state (recommended for production)
  backend "gcs" {
    bucket = var.tf_state_bucket
    prefix = var.tf_state_prefix
  }
}

# =============================================================================
# Providers
# =============================================================================

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# Configure kubernetes provider after cluster is created
provider "kubernetes" {
  host                   = "https://${google_container_cluster.blueprint.endpoint}"
  token                  = data.google_client_config.default.access_token
  cluster_ca_certificate = base64decode(google_container_cluster.blueprint.master_auth[0].cluster_ca_certificate)
}

data "google_client_config" "default" {}

# =============================================================================
# Enable Required APIs
# =============================================================================

resource "google_project_service" "apis" {
  for_each = toset([
    "container.googleapis.com",
    "containerregistry.googleapis.com",
    "artifactregistry.googleapis.com",
    "compute.googleapis.com",
    "storage.googleapis.com",
    "cloudbuild.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "cloudscheduler.googleapis.com",
    "iam.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com",
    "secretmanager.googleapis.com",
    "workflows.googleapis.com",
    "eventarc.googleapis.com",
    "run.googleapis.com",
    "pubsub.googleapis.com",
  ])

  project                    = var.project_id
  service                    = each.value
  disable_on_destroy         = false
  disable_dependent_services = false
}

# =============================================================================
# VPC Network
# =============================================================================

resource "google_compute_network" "blueprint_vpc" {
  name                    = "${var.cluster_name}-vpc"
  project                 = var.project_id
  auto_create_subnetworks = false
  routing_mode            = "REGIONAL"

  depends_on = [google_project_service.apis]
}

resource "google_compute_subnetwork" "blueprint_subnet" {
  name          = "${var.cluster_name}-subnet"
  project       = var.project_id
  region        = var.region
  network       = google_compute_network.blueprint_vpc.id
  ip_cidr_range = var.subnet_cidr

  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = var.pods_cidr
  }

  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = var.services_cidr
  }

  private_ip_google_access = true
}

# Cloud NAT for private nodes to access internet
resource "google_compute_router" "blueprint_router" {
  name    = "${var.cluster_name}-router"
  project = var.project_id
  region  = var.region
  network = google_compute_network.blueprint_vpc.id
}

resource "google_compute_router_nat" "blueprint_nat" {
  name                               = "${var.cluster_name}-nat"
  project                            = var.project_id
  router                             = google_compute_router.blueprint_router.name
  region                             = var.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"

  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}

# =============================================================================
# Service Accounts
# =============================================================================

# GKE node service account
resource "google_service_account" "gke_nodes" {
  account_id   = "${var.cluster_name}-nodes"
  display_name = "GKE Node Service Account for ${var.cluster_name}"
  project      = var.project_id
}

# Pipeline workload identity service account
resource "google_service_account" "pipeline_workload" {
  account_id   = "blueprint-pipeline-sa"
  display_name = "Blueprint Pipeline Workload Identity"
  project      = var.project_id
}

# Workflow service account
resource "google_service_account" "workflow" {
  account_id   = "blueprint-workflow-sa"
  display_name = "Blueprint Workflow Service Account"
  project      = var.project_id
}

# IAM bindings for GKE nodes
resource "google_project_iam_member" "gke_nodes_roles" {
  for_each = toset([
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/monitoring.viewer",
    "roles/storage.objectViewer",
    "roles/artifactregistry.reader",
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.gke_nodes.email}"
}

# IAM bindings for pipeline workload
resource "google_project_iam_member" "pipeline_workload_roles" {
  for_each = toset([
    "roles/storage.objectAdmin",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/aiplatform.user",
    "roles/cloudbuild.builds.editor",
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.pipeline_workload.email}"
}

# IAM bindings for workflow
resource "google_project_iam_member" "workflow_roles" {
  for_each = toset([
    "roles/workflows.invoker",
    "roles/run.invoker",
    "roles/container.developer",
    "roles/storage.objectAdmin",
    "roles/logging.logWriter",
    "roles/cloudbuild.builds.editor",
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.workflow.email}"
}

# Workload Identity binding
resource "google_service_account_iam_member" "pipeline_workload_identity" {
  service_account_id = google_service_account.pipeline_workload.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[blueprint/blueprint-pipeline-sa]"
}

# =============================================================================
# Cloud Storage
# =============================================================================

resource "google_storage_bucket" "pipeline_data" {
  name          = "${var.project_id}-blueprint-data"
  location      = var.bucket_location
  force_destroy = false
  project       = var.project_id

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  lifecycle_rule {
    condition {
      age = 365
    }
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }

  lifecycle_rule {
    condition {
      age            = 90
      matches_prefix = ["scenes/"]
      matches_suffix = ["/input/"]
    }
    action {
      type = "Delete"
    }
  }

  lifecycle_rule {
    condition {
      age            = 30
      matches_prefix = ["scenes/"]
      matches_suffix = ["/seg/", "/layout/", "/.checkpoints/"]
    }
    action {
      type = "Delete"
    }
  }

  lifecycle_rule {
    condition {
      age            = 365
      matches_prefix = ["scenes/"]
      matches_suffix = [
        "/assets/",
        "/usd/",
        "/replicator/",
        "/variation_assets/",
        "/isaac_lab/",
        "/episodes/",
      ]
    }
    action {
      type = "Delete"
    }
  }

  lifecycle_rule {
    condition {
      age            = 180
      matches_prefix = ["scenes/"]
      matches_suffix = ["/logs/", "/log/"]
    }
    action {
      type = "Delete"
    }
  }

  labels = {
    environment = var.environment
    managed_by  = "terraform"
  }

  depends_on = [google_project_service.apis]
}

# Dead letter bucket for failed jobs
resource "google_storage_bucket" "dead_letter" {
  name          = "${var.project_id}-blueprint-dead-letter"
  location      = var.bucket_location
  force_destroy = false
  project       = var.project_id

  uniform_bucket_level_access = true

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }

  labels = {
    environment = var.environment
    managed_by  = "terraform"
    purpose     = "dead-letter"
  }

  depends_on = [google_project_service.apis]
}

# =============================================================================
# Artifact Registry
# =============================================================================

resource "google_artifact_registry_repository" "blueprint_jobs" {
  location      = var.region
  repository_id = "blueprint-jobs"
  description   = "Docker images for BlueprintPipeline jobs"
  format        = "DOCKER"
  project       = var.project_id

  labels = {
    environment = var.environment
    managed_by  = "terraform"
  }

  depends_on = [google_project_service.apis]
}

# =============================================================================
# Pub/Sub for Dead Letter Queue
# =============================================================================

resource "google_pubsub_topic" "dead_letter" {
  name    = "blueprint-dead-letter"
  project = var.project_id

  labels = {
    environment = var.environment
    managed_by  = "terraform"
  }

  depends_on = [google_project_service.apis]
}

resource "google_pubsub_subscription" "dead_letter_sub" {
  name    = "blueprint-dead-letter-sub"
  topic   = google_pubsub_topic.dead_letter.name
  project = var.project_id

  ack_deadline_seconds       = 600
  message_retention_duration = "604800s" # 7 days

  retry_policy {
    minimum_backoff = "10s"
    maximum_backoff = "600s"
  }

  labels = {
    environment = var.environment
    managed_by  = "terraform"
  }
}

# =============================================================================
# Outputs
# =============================================================================

output "cluster_name" {
  value       = google_container_cluster.blueprint.name
  description = "GKE cluster name"
}

output "cluster_endpoint" {
  value       = google_container_cluster.blueprint.endpoint
  description = "GKE cluster endpoint"
  sensitive   = true
}

output "cluster_ca_certificate" {
  value       = google_container_cluster.blueprint.master_auth[0].cluster_ca_certificate
  description = "GKE cluster CA certificate"
  sensitive   = true
}

output "data_bucket" {
  value       = google_storage_bucket.pipeline_data.name
  description = "Pipeline data bucket"
}

output "dead_letter_bucket" {
  value       = google_storage_bucket.dead_letter.name
  description = "Dead letter bucket for failed jobs"
}

output "artifact_registry" {
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.blueprint_jobs.repository_id}"
  description = "Artifact Registry path for Docker images"
}

output "pipeline_service_account" {
  value       = google_service_account.pipeline_workload.email
  description = "Pipeline workload service account email"
}

output "workflow_service_account" {
  value       = google_service_account.workflow.email
  description = "Workflow service account email"
}
