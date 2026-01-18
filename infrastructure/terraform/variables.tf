# =============================================================================
# BlueprintPipeline - Terraform Variables
# =============================================================================

variable "project_id" {
  type        = string
  description = "GCP project ID"
}

variable "region" {
  type        = string
  description = "GCP region for resources"
  default     = "us-central1"
}

variable "zone" {
  type        = string
  description = "GCP zone for zonal resources"
  default     = "us-central1-a"
}

variable "enable_secondary_region" {
  type        = bool
  description = "Enable secondary region resources for regional failover"
  default     = false
}

variable "secondary_region" {
  type        = string
  description = "Secondary GCP region for failover resources"
  default     = "us-east1"
}

variable "secondary_zone" {
  type        = string
  description = "Secondary GCP zone for failover resources"
  default     = "us-east1-b"
}

variable "secondary_cluster_name" {
  type        = string
  description = "Name of the secondary GKE cluster"
  default     = "blueprint-cluster-secondary"
}

variable "bucket_location" {
  type        = string
  description = "Bucket location (multi-region or dual-region) for pipeline data"
  default     = "US"
}

variable "environment" {
  type        = string
  description = "Environment name (dev, staging, prod)"
  default     = "prod"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

# =============================================================================
# GKE Cluster Configuration
# =============================================================================

variable "cluster_name" {
  type        = string
  description = "Name of the GKE cluster"
  default     = "blueprint-cluster"
}

variable "kubernetes_version" {
  type        = string
  description = "Kubernetes version for GKE"
  default     = "1.29"  # Latest stable
}

variable "release_channel" {
  type        = string
  description = "GKE release channel"
  default     = "REGULAR"

  validation {
    condition     = contains(["RAPID", "REGULAR", "STABLE"], var.release_channel)
    error_message = "Release channel must be RAPID, REGULAR, or STABLE."
  }
}

# =============================================================================
# Node Pool Configuration
# =============================================================================

variable "cpu_node_machine_type" {
  type        = string
  description = "Machine type for CPU nodes"
  default     = "e2-standard-4"
}

variable "cpu_node_count_min" {
  type        = number
  description = "Minimum number of CPU nodes"
  default     = 1
}

variable "cpu_node_count_max" {
  type        = number
  description = "Maximum number of CPU nodes"
  default     = 10
}

variable "gpu_node_machine_type" {
  type        = string
  description = "Machine type for GPU nodes"
  default     = "n1-standard-8"
}

variable "gpu_node_count_min" {
  type        = number
  description = "Minimum number of GPU nodes"
  default     = 0
}

variable "gpu_node_count_max" {
  type        = number
  description = "Maximum number of GPU nodes"
  default     = 10
}

variable "gpu_type" {
  type        = string
  description = "Type of GPU accelerator"
  default     = "nvidia-tesla-t4"

  validation {
    condition     = contains(["nvidia-tesla-t4", "nvidia-l4", "nvidia-tesla-v100", "nvidia-a100-80gb"], var.gpu_type)
    error_message = "GPU type must be a supported NVIDIA GPU."
  }
}

variable "gpus_per_node" {
  type        = number
  description = "Number of GPUs per node"
  default     = 1
}

# =============================================================================
# GPU CI Node Pool Configuration
# =============================================================================

variable "ci_gpu_node_machine_type" {
  type        = string
  description = "Machine type for CI GPU nodes"
  default     = "n1-standard-8"
}

variable "ci_gpu_node_count_min" {
  type        = number
  description = "Minimum number of CI GPU nodes"
  default     = 0
}

variable "ci_gpu_node_count_max" {
  type        = number
  description = "Maximum number of CI GPU nodes"
  default     = 2
}

variable "ci_gpus_per_node" {
  type        = number
  description = "Number of GPUs per CI node"
  default     = 1
}

# =============================================================================
# Networking
# =============================================================================

variable "subnet_cidr" {
  type        = string
  description = "CIDR range for the subnet"
  default     = "10.0.0.0/20"
}

variable "pods_cidr" {
  type        = string
  description = "CIDR range for pods"
  default     = "10.1.0.0/16"
}

variable "services_cidr" {
  type        = string
  description = "CIDR range for services"
  default     = "10.2.0.0/20"
}

variable "master_cidr" {
  type        = string
  description = "CIDR range for GKE master"
  default     = "172.16.0.0/28"
}

variable "authorized_networks" {
  type = list(object({
    cidr_block   = string
    display_name = string
  }))
  description = "Networks authorized to access the GKE master"
  default     = []
}

# =============================================================================
# Features
# =============================================================================

variable "enable_private_nodes" {
  type        = bool
  description = "Enable private nodes (recommended for production)"
  default     = true
}

variable "enable_workload_identity" {
  type        = bool
  description = "Enable Workload Identity for GKE"
  default     = true
}

variable "enable_network_policy" {
  type        = bool
  description = "Enable network policy enforcement"
  default     = true
}

variable "enable_binary_authorization" {
  type        = bool
  description = "Enable Binary Authorization"
  default     = false
}

variable "enable_shielded_nodes" {
  type        = bool
  description = "Enable shielded GKE nodes"
  default     = true
}

# =============================================================================
# Labels and Tags
# =============================================================================

variable "labels" {
  type        = map(string)
  description = "Labels to apply to all resources"
  default = {
    managed_by = "terraform"
    project    = "blueprint-pipeline"
  }
}
