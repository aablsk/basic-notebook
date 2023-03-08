locals {
  network_name = "week2-vpc"
  notebook_name = "ddim-flowers"
  sa_notebook_name = "ddim-flowers-notebook-sa"
  sa_training_name = "ddim-flowers-training-sa"
  tensorboard_log_bucket_name = "tensorboard-logs-${google_project.project.number}"
  model_artifacts_bucket_name = "model-artifacts-${google_project.project.number}"
}