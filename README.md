# Deploy infrastructure with terraform
1. Make sure the principal you are running terraform commands with has Owner or Editor rights on the project
2. Replace values in `terraform/terraform.tfvars` with your values. `organisation_id`, `project_id` and `notebook_users` need to be replaced.
3. `terraform init`
4. `terraform apply` (you might have to run this more than once if `terraform` fails due to org policy changes taking some time to distribute)

# Clone repo
1. Start Vertex Workbench and open terminal.
2. Authenticate with your @google.com user `gcloud auth login $YOUR_LDAP@google.com` and follow instructions to authenticate.
3. Clone the repository `gcloud source repos clone rafaelsanchez-diffusion-ddim-keras-vertex --project=cloud-ce-shared-csr`
4. Revoke previous authentication `gcloud auth revoke $YOUR_LDAP@google.com`

# Training job
1. Replace values for `REGION`, `BUCKET_URI`, `SERVICE_ACCOUNT`, `PROJECT_ID` in `training/create_training_job.py
2. Navigate into training folder `cd training`
3. Create training job by running `python3 create_training_job.py`
4. Wait for training to complete and artifacts to be saved to GCS.
