# =============================================================================
# Secondary Region - Optional GKE Failover Cluster
# =============================================================================

resource "google_compute_subnetwork" "blueprint_subnet_secondary" {
  count         = var.enable_secondary_region ? 1 : 0
  name          = "${var.secondary_cluster_name}-subnet"
  project       = var.project_id
  region        = var.secondary_region
  network       = google_compute_network.blueprint_vpc.id
  ip_cidr_range = "10.10.0.0/20"

  secondary_ip_range {
    range_name    = "pods-secondary"
    ip_cidr_range = "10.11.0.0/16"
  }

  secondary_ip_range {
    range_name    = "services-secondary"
    ip_cidr_range = "10.12.0.0/20"
  }

  private_ip_google_access = true
}

resource "google_compute_router" "blueprint_router_secondary" {
  count   = var.enable_secondary_region ? 1 : 0
  name    = "${var.secondary_cluster_name}-router"
  project = var.project_id
  region  = var.secondary_region
  network = google_compute_network.blueprint_vpc.id
}

resource "google_compute_router_nat" "blueprint_nat_secondary" {
  count                              = var.enable_secondary_region ? 1 : 0
  name                               = "${var.secondary_cluster_name}-nat"
  project                            = var.project_id
  router                             = google_compute_router.blueprint_router_secondary[0].name
  region                             = var.secondary_region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"

  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}

resource "google_container_cluster" "blueprint_secondary" {
  count    = var.enable_secondary_region ? 1 : 0
  name     = var.secondary_cluster_name
  location = var.secondary_zone
  project  = var.project_id

  remove_default_node_pool = true
  initial_node_count       = 1

  network    = google_compute_network.blueprint_vpc.name
  subnetwork = google_compute_subnetwork.blueprint_subnet_secondary[0].name

  ip_allocation_policy {
    cluster_secondary_range_name  = "pods-secondary"
    services_secondary_range_name = "services-secondary"
  }

  private_cluster_config {
    enable_private_nodes    = var.enable_private_nodes
    enable_private_endpoint = false
    master_ipv4_cidr_block  = "172.16.1.0/28"
  }

  release_channel {
    channel = var.release_channel
  }

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

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

  resource_labels = merge(var.labels, {
    environment = var.environment
    region      = var.secondary_region
  })

  depends_on = [
    google_project_service.apis,
    google_compute_subnetwork.blueprint_subnet_secondary,
    google_service_account.gke_nodes,
  ]
}

resource "google_container_node_pool" "secondary_cpu_pool" {
  count    = var.enable_secondary_region ? 1 : 0
  name     = "cpu-pool"
  location = var.secondary_zone
  cluster  = google_container_cluster.blueprint_secondary[0].name
  project  = var.project_id

  autoscaling {
    min_node_count  = var.cpu_node_count_min
    max_node_count  = var.cpu_node_count_max
    location_policy = "BALANCED"
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

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
      region    = var.secondary_region
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
}

resource "google_container_node_pool" "secondary_gpu_pool" {
  count    = var.enable_secondary_region ? 1 : 0
  name     = "gpu-pool"
  location = var.secondary_zone
  cluster  = google_container_cluster.blueprint_secondary[0].name
  project  = var.project_id

  autoscaling {
    min_node_count  = var.gpu_node_count_min
    max_node_count  = var.gpu_node_count_max
    location_policy = "ANY"
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    machine_type = var.gpu_node_machine_type
    disk_size_gb = 200
    disk_type    = "pd-ssd"

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
      region    = var.secondary_region
    }

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
  }
}

output "secondary_cluster_name" {
  value       = var.enable_secondary_region ? google_container_cluster.blueprint_secondary[0].name : ""
  description = "Secondary GKE cluster name"
}

output "secondary_cluster_location" {
  value       = var.enable_secondary_region ? google_container_cluster.blueprint_secondary[0].location : ""
  description = "Secondary GKE cluster location"
}
