resource "google_service_account" "scheduler" {
    project = var.project_id
    account_id = "scheduler"
}

resource "google_cloud_scheduler_job" "daily_pipeline_run" {
  name             = "daily-pipeline-run"
  description      = "run vertex pipeline and deploy model if conditions met"
  schedule         = "0 5 * * *"
  time_zone        = "Europe/London"
  attempt_deadline = "320s"

  retry_config {
    retry_count = 1
  }

  http_target {
    http_method = "POST"
    uri         = google_cloudfunctions_function.trigger_func.https_trigger_url
    body        = base64encode("{\"parameter_values\":{\"bq_table_uri\":\"bq://ml-coaching-team95-week3.bank_data.raw_bank_marketing_data\",\"num_units\":64,\"epochs\":3,\"deploy_model\":\"True\"}}")

    oidc_token {
        service_account_email = google_service_account.scheduler.email
    }
  }
}

resource "google_cloudfunctions_function_iam_member" "trigger_func_scheduler" {
  project        = var.project_id
  region         = var.region
  cloud_function = google_cloudfunctions_function.trigger_func.name
  role           = "roles/cloudfunctions.invoker"
  member         = "serviceAccount:${google_service_account.scheduler.email}"
}
