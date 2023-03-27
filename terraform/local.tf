locals {
  network_name = "notebook-vpc"
  notebook_name = "week3"
  sa_notebook_name = "week3-notebook-sa"
  sa_training_name = "week3-training-sa"
  sa_serving_name = "week3-serving-sa"
  training_bucket_name = "week3-training-${data.google_project.project.number}"
  sa_trigger_func_name = "week3-trigger-func-sa"
}