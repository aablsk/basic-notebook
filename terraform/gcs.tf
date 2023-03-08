resource "google_service_account" "training" {
  project    = var.project_id
  account_id = local.sa_training_name

  depends_on = [
    module.enabled_google_apis
  ]
}

resource "google_storage_bucket" "tensorboard_logs" {
  project                     = var.project_id
  location                    = var.region
  name                        = local.tensorboard_log_bucket_name
  uniform_bucket_level_access = true

  depends_on = [
    module.enabled_google_apis
  ]
}

resource "google_storage_bucket" "model_artifacts" {
    project                     = var.project_id
  location                    = var.region
  name                        = local.model_artifacts_bucket_name
  uniform_bucket_level_access = true

  depends_on = [
    module.enabled_google_apis
  ]
}

resource "google_storage_bucket_iam_member" "tensorboard_logs" {
  bucket  = google_storage_bucket.tensorboard_logs.name

  member = "serviceAccount:${google_service_account.training.email}"
  role   = "roles/storage.admin"
}

resource "google_storage_bucket_iam_member" "tensorboard_logs" {
  bucket  = google_storage_bucket.model_artifacts.name

  member = "serviceAccount:${google_service_account.training.email}"
  role   = "roles/storage.admin"
}
