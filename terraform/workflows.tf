# Data sources for existing workflows
# These workflows should already be deployed from the workflows/ directory

data "google_workflows_workflow" "hunyuan_pipeline" {
  name     = "hunyuan-pipeline"
  location = var.region
}

data "google_workflows_workflow" "sam3d_pipeline" {
  name     = "sam3d-pipeline"
  location = var.region
}

data "google_workflows_workflow" "interactive_pipeline" {
  name     = "interactive-pipeline"
  location = var.region
}

data "google_workflows_workflow" "usd_assembly_pipeline" {
  name     = "usd-assembly-pipeline"
  location = var.region
}

# If workflows don't exist yet, you can create them here:
# Uncomment and use these if you need to create the workflows via Terraform

# resource "google_workflows_workflow" "hunyuan_pipeline" {
#   name            = "hunyuan-pipeline"
#   region          = var.region
#   source_contents = file("${path.module}/../workflows/hunyuan-pipeline.yaml")
#   service_account = var.service_account_email
# }

# resource "google_workflows_workflow" "sam3d_pipeline" {
#   name            = "sam3d-pipeline"
#   region          = var.region
#   source_contents = file("${path.module}/../workflows/sam3d-pipeline.yaml")
#   service_account = var.service_account_email
# }

# resource "google_workflows_workflow" "interactive_pipeline" {
#   name            = "interactive-pipeline"
#   region          = var.region
#   source_contents = file("${path.module}/../workflows/interactive-pipeline.yaml")
#   service_account = var.service_account_email
# }

# resource "google_workflows_workflow" "usd_assembly_pipeline" {
#   name            = "usd-assembly-pipeline"
#   region          = var.region
#   source_contents = file("${path.module}/../workflows/usd-assembly-pipeline.yaml")
#   service_account = var.service_account_email
# }
