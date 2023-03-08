module "vpc" {
  source  = "terraform-google-modules/network/google"
  version = "~> 6.0"

  project_id   = var.project_id
  network_name = local.network_name
  routing_mode = "REGIONAL"

  subnets = [
    {
      subnet_name           = "workbench-subnet"
      subnet_ip             = "10.10.10.0/28"
      subnet_region         = var.region
      subnet_private_access = "true"
    }
  ]

  depends_on = [
    module.enabled_google_apis
  ]
}
