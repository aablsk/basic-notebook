# Deploy infrastructure with terraform
1. Connect your GitHub Repo to CloudBuild in the region you want to deploy your resources to.
1. Make sure the principal you are running terraform commands with has Owner or Editor rights on the project
1. Replace values in `terraform/terraform.tfvars` with your values. `organisation_id`, `project_id` and `notebook_users` need to be replaced.
1. `terraform init`
1. `terraform apply` (you might have to run this more than once if `terraform` fails due to org policy changes taking some time to distribute)
