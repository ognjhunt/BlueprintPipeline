# =============================================================================
# GKE Cluster Configuration
# =============================================================================

resource "google_container_cluster" "blueprint" {
  name     = var.cluster_name
  location = var.zone
  project  = var.project_id

  # We manage node pools separately
  remove_default_node_pool = true
  initial_node_count       = 1

  # Networking
  network    = google_compute_network.blueprint_vpc.name
  subnetwork = google_compute_subnetwork.blueprint_subnet.name

  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  # Private cluster configuration
  private_cluster_config {
    enable_private_nodes    = var.enable_private_nodes
    enable_private_endpoint = false  # Allow public access to master
    master_ipv4_cidr_block  = var.master_cidr
  }

  # Master authorized networks
  master_authorized_networks_config {
    dynamic "cidr_blocks" {
      for_each = var.master_authorized_networks
      content {
        cidr_block   = cidr_blocks.value.cidr_block
        display_name = cidr_blocks.value.display_name
      }
    }
  }

  # Release channel
  release_channel {
    channel = var.release_channel
  }

  # Workload Identity
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  # Addons
  addons_config {
    http_load_balancing {
      disabled = false
    }

    horizontal_pod_autoscaling {
      disabled = false
    }

    network_policy_config {
      disabled = !var.enable_network_policy
    }

    gce_persistent_disk_csi_driver_config {
      enabled = true
    }

    gcs_fuse_csi_driver_config {
      enabled = true
    }
  }

  # Network policy
  network_policy {
    enabled  = var.enable_network_policy
    provider = var.enable_network_policy ? "CALICO" : "PROVIDER_UNSPECIFIED"
  }

  # Monitoring and logging
  logging_config {
    enable_components = ["SYSTEM_COMPONENTS", "WORKLOADS"]
  }

  monitoring_config {
    enable_components = ["SYSTEM_COMPONENTS"]

    managed_prometheus {
      enabled = true
    }
  }

  # Cluster autoscaling (for GPU nodes)
  cluster_autoscaling {
    enabled = true

    resource_limits {
      resource_type = "cpu"
      minimum       = 4
      maximum       = 200
    }

    resource_limits {
      resource_type = "memory"
      minimum       = 16
      maximum       = 800
    }

    resource_limits {
      resource_type = "nvidia-tesla-t4"
      minimum       = 0
      maximum       = var.gpu_node_count_max
    }

    auto_provisioning_defaults {
      oauth_scopes = [
        "https://www.googleapis.com/auth/cloud-platform",
      ]

      service_account = google_service_account.gke_nodes.email

      management {
        auto_repair  = true
        auto_upgrade = true
      }

      shielded_instance_config {
        enable_secure_boot          = var.enable_shielded_nodes
        enable_integrity_monitoring = var.enable_shielded_nodes
      }
    }
  }

  # Maintenance policy (off-peak hours)
  maintenance_policy {
    recurring_window {
      start_time = "2024-01-01T03:00:00Z"
      end_time   = "2024-01-01T07:00:00Z"
      recurrence = "FREQ=WEEKLY;BYDAY=SA,SU"
    }
  }

  # Binary authorization
  dynamic "binary_authorization" {
    for_each = var.enable_binary_authorization ? [1] : []
    content {
      evaluation_mode = "PROJECT_SINGLETON_POLICY_ENFORCE"
    }
  }

  # Cost management
  cost_management_config {
    enabled = true
  }

  # Node config defaults
  node_config {
    shielded_instance_config {
      enable_secure_boot          = var.enable_shielded_nodes
      enable_integrity_monitoring = var.enable_shielded_nodes
    }
  }

  resource_labels = merge(var.labels, {
    environment = var.environment
  })

  depends_on = [
    google_project_service.apis,
    google_compute_subnetwork.blueprint_subnet,
    google_service_account.gke_nodes,
  ]

  lifecycle {
    ignore_changes = [
      # Ignore changes to node_config as we manage node pools separately
      node_config,
    ]
  }
}

# =============================================================================
# CPU Node Pool (for non-GPU workloads)
# =============================================================================

resource "google_container_node_pool" "cpu_pool" {
  name     = "cpu-pool"
  location = var.zone
  cluster  = google_container_cluster.blueprint.name
  project  = var.project_id

  # Autoscaling
  autoscaling {
    min_node_count  = var.cpu_node_count_min
    max_node_count  = var.cpu_node_count_max
    location_policy = "BALANCED"
  }

  # Node management
  management {
    auto_repair  = true
    auto_upgrade = true
  }

  # Node configuration
  node_config {
    machine_type = var.cpu_node_machine_type
    disk_size_gb = 100
    disk_type    = "pd-balanced"

    service_account = google_service_account.gke_nodes.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]

    labels = {
      node_type = "cpu"
    }

    metadata = {
      disable-legacy-endpoints = "true"
    }

    shielded_instance_config {
      enable_secure_boot          = var.enable_shielded_nodes
      enable_integrity_monitoring = var.enable_shielded_nodes
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }

  lifecycle {
    ignore_changes = [
      node_config[0].labels,
    ]
  }
}

# =============================================================================
# GPU Node Pool (for Isaac Sim workloads)
# =============================================================================

resource "google_container_node_pool" "gpu_pool" {
  name     = "gpu-pool"
  location = var.zone
  cluster  = google_container_cluster.blueprint.name
  project  = var.project_id

  # Autoscaling - start at 0 to save costs
  autoscaling {
    min_node_count  = var.gpu_node_count_min
    max_node_count  = var.gpu_node_count_max
    location_policy = "ANY"
  }

  # Node management
  management {
    auto_repair  = true
    auto_upgrade = true
  }

  # Node configuration
  node_config {
    machine_type = var.gpu_node_machine_type
    disk_size_gb = 200  # Larger disk for Isaac Sim cache
    disk_type    = "pd-ssd"

    # GPU configuration
    guest_accelerator {
      type  = var.gpu_type
      count = var.gpus_per_node

      gpu_driver_installation_config {
        gpu_driver_version = "DEFAULT"
      }
    }

    service_account = google_service_account.gke_nodes.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]

    labels = {
      node_type = "gpu"
      gpu_type  = var.gpu_type
    }

    # Taint GPU nodes so only GPU workloads get scheduled
    taint {
      key    = "nvidia.com/gpu"
      value  = "present"
      effect = "NO_SCHEDULE"
    }

    metadata = {
      disable-legacy-endpoints = "true"
    }

    shielded_instance_config {
      enable_secure_boot          = var.enable_shielded_nodes
      enable_integrity_monitoring = var.enable_shielded_nodes
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    # Preemptible for cost savings (remove for production)
    # spot = true
  }

  lifecycle {
    ignore_changes = [
      node_config[0].labels,
    ]
  }
}

# =============================================================================
# GPU CI Node Pool (for GitHub Actions runners)
# =============================================================================

resource "google_container_node_pool" "gpu_ci_pool" {
  count    = var.ci_gpu_node_count_max > 0 ? 1 : 0
  name     = "gpu-ci-pool"
  location = var.zone
  cluster  = google_container_cluster.blueprint.name
  project  = var.project_id

  autoscaling {
    min_node_count  = var.ci_gpu_node_count_min
    max_node_count  = var.ci_gpu_node_count_max
    location_policy = "ANY"
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    machine_type = var.ci_gpu_node_machine_type
    disk_size_gb = 200
    disk_type    = "pd-ssd"

    guest_accelerator {
      type  = var.gpu_type
      count = var.ci_gpus_per_node

      gpu_driver_installation_config {
        gpu_driver_version = "DEFAULT"
      }
    }

    service_account = google_service_account.gke_nodes.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]

    labels = {
      node_type = "gpu-ci"
      gpu_type  = var.gpu_type
      workload  = "github-actions"
    }

    taint {
      key    = "github-actions"
      value  = "ci"
      effect = "NO_SCHEDULE"
    }

    metadata = {
      disable-legacy-endpoints = "true"
    }

    shielded_instance_config {
      enable_secure_boot          = var.enable_shielded_nodes
      enable_integrity_monitoring = var.enable_shielded_nodes
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }

  lifecycle {
    ignore_changes = [
      node_config[0].labels,
    ]
  }
}

# =============================================================================
# Kubernetes Resources
# =============================================================================

# Blueprint namespace
resource "kubernetes_namespace" "blueprint" {
  metadata {
    name = "blueprint"

    labels = {
      environment = var.environment
      managed_by  = "terraform"
    }
  }

  depends_on = [google_container_cluster.blueprint]
}

# Service account for workload identity
resource "kubernetes_service_account" "pipeline" {
  metadata {
    name      = "blueprint-pipeline-sa"
    namespace = kubernetes_namespace.blueprint.metadata[0].name

    annotations = {
      "iam.gke.io/gcp-service-account" = google_service_account.pipeline_workload.email
    }

    labels = {
      environment = var.environment
      managed_by  = "terraform"
    }
  }

  depends_on = [
    kubernetes_namespace.blueprint,
    google_service_account_iam_member.pipeline_workload_identity,
  ]
}

# Firebase service account secret (mounted by workflow jobs)
resource "kubernetes_secret" "firebase_service_account" {
  count = var.firebase_service_account_json != "" ? 1 : 0

  metadata {
    name      = "firebase-service-account"
    namespace = kubernetes_namespace.blueprint.metadata[0].name

    labels = {
      environment = var.environment
      managed_by  = "terraform"
    }
  }

  type = "Opaque"

  string_data = {
    "service-account.json" = var.firebase_service_account_json
  }

  depends_on = [kubernetes_namespace.blueprint]
}

# Resource quotas
resource "kubernetes_resource_quota" "blueprint" {
  metadata {
    name      = "blueprint-quota"
    namespace = kubernetes_namespace.blueprint.metadata[0].name
  }

  spec {
    hard = {
      "requests.cpu"            = "100"
      "requests.memory"         = "200Gi"
      "limits.cpu"              = "200"
      "limits.memory"           = "400Gi"
      "requests.nvidia.com/gpu" = "20"
      "limits.nvidia.com/gpu"   = "20"
      "pods"                    = "100"
    }
  }

  depends_on = [kubernetes_namespace.blueprint]
}

# Limit ranges
resource "kubernetes_limit_range" "blueprint" {
  metadata {
    name      = "blueprint-limits"
    namespace = kubernetes_namespace.blueprint.metadata[0].name
  }

  spec {
    limit {
      type = "Container"

      default = {
        cpu    = "2"
        memory = "4Gi"
      }

      default_request = {
        cpu    = "500m"
        memory = "1Gi"
      }
    }
  }

  depends_on = [kubernetes_namespace.blueprint]
}

# NVIDIA GPU device plugin (if not using GKE managed driver)
resource "kubernetes_daemonset" "nvidia_device_plugin" {
  metadata {
    name      = "nvidia-device-plugin"
    namespace = "kube-system"
  }

  spec {
    selector {
      match_labels = {
        name = "nvidia-device-plugin"
      }
    }

    template {
      metadata {
        labels = {
          name = "nvidia-device-plugin"
        }
      }

      spec {
        toleration {
          key      = "nvidia.com/gpu"
          operator = "Exists"
          effect   = "NoSchedule"
        }

        priority_class_name = "system-node-critical"

        container {
          name  = "nvidia-device-plugin"
          image = "nvcr.io/nvidia/k8s-device-plugin:v0.14.3"

          security_context {
            privileged = true
          }

          volume_mount {
            name       = "device-plugin"
            mount_path = "/var/lib/kubelet/device-plugins"
          }
        }

        volume {
          name = "device-plugin"
          host_path {
            path = "/var/lib/kubelet/device-plugins"
          }
        }

        node_selector = {
          "cloud.google.com/gke-accelerator" = var.gpu_type
        }
      }
    }
  }

  depends_on = [google_container_node_pool.gpu_pool]
}
