resource "google_service_account" "training" {
  project    = var.project_id
  account_id = local.sa_training_name

  depends_on = [
    module.enabled_google_apis
  ]
}

resource "google_storage_bucket" "training" {
  project                     = var.project_id
  location                    = var.region
  name                        = local.training_bucket_name
  uniform_bucket_level_access = true

  depends_on = [
    module.enabled_google_apis
  ]
}

resource "google_storage_bucket_iam_member" "training" {
  bucket = google_storage_bucket.training.name

  member = "serviceAccount:${google_service_account.training.email}"
  role   = "roles/storage.admin"
}
