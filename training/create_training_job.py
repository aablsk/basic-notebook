import os
from datetime import datetime as dt
from google.cloud import aiplatform
from google.cloud import storage

# CONSTANTS
# TODO: update this with your region
REGION = "us-central1"
# TODO: update this with your bucket name
BUCKET_URI = "gs://ddim-flowers-training-665218981807"
# TODO: update this with your service account
SERVICE_ACCOUNT = "ddim-flowers-training-sa@mlcoaching-week2-aablsk.iam.gserviceaccount.com"
# TODO: update this with your project_id
PROJECT_ID = "mlcoaching-week2-aablsk"
# TODO: INCREASE FOR BETTER RESULTS
NUM_EPOCHS = 50
TRAINING_IMAGE_URI = "us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-11:latest"

# DYNAMIC ENV VARS
JOB_ID = dt.now().strftime('%Y_%m_%d__%H_%M_%S')
JOB_BUCKET_URI = os.path.join(BUCKET_URI, JOB_ID)

aiplatform.init(project=PROJECT_ID, location=REGION, experiment="ddim-flowers")

training_job = aiplatform.CustomTrainingJob(
    project=PROJECT_ID,
    location=REGION,
    display_name=f"ddim-flowers-training-{JOB_ID}",
    container_uri=TRAINING_IMAGE_URI,
    script_path="train_ddim.py",
    staging_bucket=JOB_BUCKET_URI,
    model_description="DDIM finetuned on oxford_flowers102 dataset",
    labels={
        "num_epochs": str(NUM_EPOCHS),
        "job_id": JOB_ID
    },
    model_serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-11:latest"
)

training_job.run(
    model_display_name="ddim_flowers",
    model_labels={
        "num_epochs": str(NUM_EPOCHS),
        "job_id": JOB_ID
    },
    machine_type="n1-standard-8",
    replica_count=1,
    environment_variables={
        "NUM_EPOCHS": str(NUM_EPOCHS)
    },
    service_account=SERVICE_ACCOUNT,
    accelerator_type="NVIDIA_TESLA_V100",
    accelerator_count=1,
    sync=False,
)
