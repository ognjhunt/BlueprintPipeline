variable "project_id" {
  description = "GCP Project ID"
  type        = string
  default     = "blueprint-8c1ca"
}

variable "region" {
  description = "GCP region for resources"
  type        = string
  default     = "us-central1"
}

variable "bucket_name" {
  description = "GCS bucket name for pipeline data"
  type        = string
  # Set this to your actual bucket name
  # default     = "your-bucket-name"
}

variable "service_account_email" {
  description = "Service account email for Eventarc triggers"
  type        = string
  default     = "744608654760-compute@developer.gserviceaccount.com"
}
