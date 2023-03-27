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

resource "google_project_iam_member" "cloudbuild_logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.cloudbuild.email}"
}

resource "google_service_account_iam_member" "training_sa_user_cloudbuild" {

  service_account_id = google_service_account.training.id
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.cloudbuild.email}"
}

resource "google_project_iam_member" "cloudbuild_vertex_admin" {
  project = var.project_id
  role    = "roles/aiplatform.admin"
  member  = "serviceAccount:${google_service_account.cloudbuild.email}"
}

resource "google_project_iam_member" "cloudbuild_trigger_func" {
  project        = var.project_id
  role           = "roles/cloudfunctions.developer"
  member         = "serviceAccount:${google_service_account.cloudbuild.email}"
}
