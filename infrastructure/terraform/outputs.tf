# =============================================================================
# BlueprintPipeline - Terraform Outputs
# =============================================================================

output "gke_cluster_name" {
  value       = google_container_cluster.blueprint.name
  description = "The name of the GKE cluster"
}

output "gke_cluster_location" {
  value       = google_container_cluster.blueprint.location
  description = "The location of the GKE cluster"
}

output "gke_cluster_endpoint" {
  value       = google_container_cluster.blueprint.endpoint
  description = "The endpoint for the GKE cluster"
  sensitive   = true
}

output "gke_cluster_ca_certificate" {
  value       = google_container_cluster.blueprint.master_auth[0].cluster_ca_certificate
  description = "The CA certificate for the GKE cluster"
  sensitive   = true
}

output "gke_get_credentials_command" {
  value       = "gcloud container clusters get-credentials ${google_container_cluster.blueprint.name} --zone ${google_container_cluster.blueprint.location} --project ${var.project_id}"
  description = "Command to get GKE credentials"
}

# Storage outputs
output "pipeline_bucket_name" {
  value       = google_storage_bucket.pipeline_data.name
  description = "The name of the pipeline data bucket"
}

output "pipeline_bucket_url" {
  value       = google_storage_bucket.pipeline_data.url
  description = "The URL of the pipeline data bucket"
}

output "dead_letter_bucket_name" {
  value       = google_storage_bucket.dead_letter.name
  description = "The name of the dead letter bucket"
}

# Artifact Registry outputs
output "artifact_registry_repository" {
  value       = google_artifact_registry_repository.blueprint_jobs.repository_id
  description = "The Artifact Registry repository ID"
}

output "artifact_registry_location" {
  value       = google_artifact_registry_repository.blueprint_jobs.location
  description = "The Artifact Registry location"
}

output "docker_push_command" {
  value       = "docker push ${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.blueprint_jobs.repository_id}/IMAGE:TAG"
  description = "Command to push Docker images to Artifact Registry"
}

output "docker_auth_command" {
  value       = "gcloud auth configure-docker ${var.region}-docker.pkg.dev"
  description = "Command to configure Docker authentication for Artifact Registry"
}

# Service account outputs
output "gke_nodes_service_account" {
  value       = google_service_account.gke_nodes.email
  description = "Service account for GKE nodes"
}

output "pipeline_service_account" {
  value       = google_service_account.pipeline_workload.email
  description = "Service account for pipeline workloads"
}

output "workflow_service_account" {
  value       = google_service_account.workflow.email
  description = "Service account for Cloud Workflows"
}

# Networking outputs
output "vpc_network_name" {
  value       = google_compute_network.blueprint_vpc.name
  description = "The VPC network name"
}

output "subnet_name" {
  value       = google_compute_subnetwork.blueprint_subnet.name
  description = "The subnet name"
}

# Pub/Sub outputs
output "dead_letter_topic" {
  value       = google_pubsub_topic.dead_letter.name
  description = "Dead letter Pub/Sub topic"
}

output "dead_letter_subscription" {
  value       = google_pubsub_subscription.dead_letter_sub.name
  description = "Dead letter Pub/Sub subscription"
}

# Node pool outputs
output "cpu_node_pool" {
  value       = google_container_node_pool.cpu_pool.name
  description = "CPU node pool name"
}

output "gpu_node_pool" {
  value       = google_container_node_pool.gpu_pool.name
  description = "GPU node pool name"
}

output "gpu_ci_node_pool" {
  value       = try(google_container_node_pool.gpu_ci_pool[0].name, null)
  description = "GPU CI node pool name"
}

output "gpu_type" {
  value       = var.gpu_type
  description = "GPU type in the GPU node pool"
}

# Summary for quick reference
output "quick_start_summary" {
  value = <<-EOT

    ============================================================
    BlueprintPipeline Infrastructure Ready!
    ============================================================

    GKE Cluster: ${google_container_cluster.blueprint.name}
    Location:    ${google_container_cluster.blueprint.location}
    Project:     ${var.project_id}

    Get credentials:
      ${format("gcloud container clusters get-credentials %s --zone %s --project %s", google_container_cluster.blueprint.name, google_container_cluster.blueprint.location, var.project_id)}

    Configure Docker:
      gcloud auth configure-docker ${var.region}-docker.pkg.dev

    Push images:
      docker push ${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.blueprint_jobs.repository_id}/IMAGE:TAG

    Data bucket:
      gs://${google_storage_bucket.pipeline_data.name}/

    Deploy workloads:
      kubectl apply -f k8s/

    ============================================================
  EOT

  description = "Quick start summary"
}
