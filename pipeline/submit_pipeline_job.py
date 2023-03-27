from google.cloud import aiplatform
from datetime import datetime

PROJECT_ID = "ml-coaching-team95-week3"  # @param {type:"string"}
REGION = "us-central1"  # @param {type: "string"}
    
BUCKET_NAME = "week3-training-124142000173"
PIPELINE_DEFINITION_BUCKET_NAME = "vertexai_pipeline_definition_ml-coaching-team95-week3"
TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")

EXPERIMENT_NAME = f"mlops-part-one-experimet-{TIMESTAMP}"
EXPERIMENT_DESCRIPTION = "Running experiments in our first MLOps Pipeline"

PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipeline_root/mlops-coaching-part-one"

aiplatform.init(project=PROJECT_ID, 
         staging_bucket=BUCKET_NAME,
         experiment=EXPERIMENT_NAME,
         experiment_description=EXPERIMENT_DESCRIPTION)

job = aiplatform.PipelineJob(
        display_name="pipeline_run_with_data_validation", #todo - add display  name,
        template_path=f"gs://{PIPELINE_DEFINITION_BUCKET_NAME}/pipeline.json", #todo - COMPILED_PIPELINE_PACKAGE_PATH,
        pipeline_root=PIPELINE_ROOT, #todo - add pipeline root defined above, which is the GCS path where the pipeline artifacts will be stored,
        parameter_values={
            # you can modify your pipeline values here e.g.
            "bq_table_uri" : "bq://ml-coaching-team95-week3.bank_data.raw_bank_marketing_data", ## Add the bq table uri of the raw data: e.g. bq://my_project.mydataset.mytable',
            "num_units": 64,
            "epochs": 3
        },
    )

job.submit(
    service_account="week3-training-sa@ml-coaching-team95-week3.iam.gserviceaccount.com"
)
