terraform {
}

# google-beta provider for fleet API, artifact registry
provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# default google provider for most resources
provider "google" {
  project = var.project_id
  region  = var.region
}

data "google_project" "project" {
}