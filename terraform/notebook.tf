resource "google_notebooks_instance" "notebook" {
  project  = var.project_id
  name     = local.notebook_name
  location = var.zone
  # access
  service_account = google_service_account.notebook.email
  # networking
  network = module.vpc.network_id
  subnet  = module.vpc.subnets_ids[0]
  # machine configuraiton
  machine_type       = "n1-standard-8"
  boot_disk_type     = "PD_SSD"
  boot_disk_size_gb  = 100
  install_gpu_driver = true
  vm_image {
    project      = "deeplearning-platform-release"
    image_family = "tf-2-11-cu113-notebooks-debian-10"
  }
  accelerator_config {
    core_count = 1
    type       = "NVIDIA_TESLA_T4"
  }

  no_public_ip    = false
  no_proxy_access = false

  depends_on = [
    google_project_organization_policy.requiredShieldedVM,
    google_project_organization_policy.vmExternalIpAccess
  ]
}

resource "google_service_account" "notebook" {
  project    = var.project_id
  account_id = local.sa_notebook_name

  depends_on = [
    module.enabled_google_apis
  ]
}

resource "google_service_account_iam_member" "notebook_users" {
  for_each = toset(var.notebook_users)

  service_account_id = google_service_account.notebook.id
  role               = "roles/iam.serviceAccountUser"
  member             = each.value
}

resource "google_project_iam_member" "notebook_vertex_admin" {
  project = var.project_id
  role    = "roles/aiplatform.admin"
  member  = "serviceAccount:${google_service_account.notebook.email}"
}


resource "google_project_iam_member" "notebook_bq_admin" {
  project = var.project_id
  role    = "roles/bigquery.admin"
  member  = "serviceAccount:${google_service_account.notebook.email}"
}

resource "google_storage_bucket_iam_member" "notebook" {
  bucket = google_storage_bucket.training.name

  member = "serviceAccount:${google_service_account.notebook.email}"
  role   = "roles/storage.admin"
}
