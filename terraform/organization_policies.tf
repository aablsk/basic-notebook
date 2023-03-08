resource "google_project_organization_policy" "requiredShieldedVM" {
  project    = var.project_id
  constraint = "compute.requireShieldedVm"

  restore_policy {
    default = true
  }
}

resource "google_project_organization_policy" "vmExternalIpAccess" {
  project    = var.project_id
  constraint = "compute.vmExternalIpAccess"

  restore_policy {
    default = true
  }
}
