resource "google_service_account" "cloudbuild" {
  project    = var.project_id
  account_id = "cloudbuild"
}

# CI trigger configuration for rendering pipeline
resource "google_cloudbuild_trigger" "render-pipeline" {
  project  = var.project_id
  name     = "render-pipeline-definition"
  location = var.region

  github {
    owner = var.repo_owner
    name  = var.repo_name

    push {
      branch = "^week4$"
    }
  }

  included_files = ["pipeline/**", "cloudbuild.yaml"]
  filename       = "pipeline/cloudbuild.yaml"
  substitutions = {
    _PIPELINE_BUCKET = "gs://${google_storage_bucket.pipeline_definition.name}"
  }
  service_account = google_service_account.cloudbuild.id
}

# give CloudBuild SA access to read pipeline definition
resource "google_storage_bucket_iam_member" "pipeline_definition_cloudbuild" {
  bucket = google_storage_bucket.pipeline_definition.name

  member = "serviceAccount:${google_service_account.cloudbuild.email}"
  role   = "roles/storage.admin"
}

# bucket to store vertex ai pipeline definition
resource "google_storage_bucket" "pipeline_definition" {
  project                     = var.project_id
  name                        = "vertexai_pipeline_definition_${var.project_id}"
  uniform_bucket_level_access = true
  location                    = var.region
  force_destroy               = true # TODO: remove this so buckets don't get deleted on terraform destroy

  versioning {
    enabled = true
  }
}

# give SA access to read pipeline definition
resource "google_storage_bucket_iam_member" "pipeline_definition" {
  bucket = google_storage_bucket.pipeline_definition.name

  member = "serviceAccount:${google_service_account.training.email}"
  role   = "roles/storage.objectViewer"
}
