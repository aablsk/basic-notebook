resource "google_service_account" "trigger_func" {
  project    = var.project_id
  account_id = local.sa_trigger_func_name

  depends_on = [
    module.enabled_google_apis
  ]
}

resource "google_cloudfunctions_function" "trigger_func" {
  name        = "trigger_func"
  description = "Triggers the Vertex AI pipeline"
  runtime     = "python39"

  trigger_http = true
  entry_point  = "process_request"

  source_archive_bucket = google_storage_bucket.trigger_func_source.name
  source_archive_object = google_storage_bucket_object.archive.name

  service_account_email = google_service_account.trigger_func.email

  ingress_settings = "ALLOW_INTERNAL_AND_GCLB"
}

resource "google_project_iam_member" "trigger_func_logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.trigger_func.email}"
}

resource "google_service_account_iam_member" "training_sa_user_trigger_func" {

  service_account_id = google_service_account.training.id
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.trigger_func.email}"
}

resource "google_project_iam_member" "trigger_func_vertex_admin" {
  project = var.project_id
  role    = "roles/aiplatform.admin"
  member  = "serviceAccount:${google_service_account.trigger_func.email}"
}

resource "google_storage_bucket" "trigger_func_source" {
  name     = "trigger_func_source"
  location = var.region
  uniform_bucket_level_access = true
}

resource "google_storage_bucket_object" "archive" {
  name   = "index.zip"
  bucket = google_storage_bucket.trigger_func_source.name
  source = "../pipeline/pipeline_trigger_function/pipeline_trigger_func_src.zip"
}

resource "google_storage_bucket_iam_member" "pipeline_definition_trigger_func" {
  bucket = google_storage_bucket.pipeline_definition.name

  member = "serviceAccount:${google_service_account.trigger_func.email}"
  role   = "roles/storage.objectViewer"
}
