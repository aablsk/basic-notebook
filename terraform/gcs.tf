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
