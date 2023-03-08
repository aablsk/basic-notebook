locals {
  network_name = "week2-vpc"
  notebook_name = "ddim-flowers"
  sa_notebook_name = "ddim-flowers-notebook-sa"
  sa_training_name = "ddim-flowers-training-sa"
  training_bucket_name = "ddim-flowers-training-${data.google_project.project.number}"
}