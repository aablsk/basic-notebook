module "enabled_google_apis" {
  source  = "terraform-google-modules/project-factory/google//modules/project_services"
  version = "~> 13.0"

  project_id                  = var.project_id
  disable_services_on_destroy = false

  activate_apis = [
    "aiplatform.googleapis.com",
    "compute.googleapis.com",
    "dns.googleapis.com",
    "iam.googleapis.com",
    "notebooks.googleapis.com",
    "artifactregistry.googleapis.com"
  ]
}