resource "google_service_account" "serving" {
  project    = var.project_id
  account_id = local.sa_serving_name

  depends_on = [
    module.enabled_google_apis
  ]
}

resource "google_storage_bucket_iam_member" "serving" {
  bucket = google_storage_bucket.training.name

  member = "serviceAccount:${google_service_account.serving.email}"
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
