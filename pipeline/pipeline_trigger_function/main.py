import json
from google.cloud import aiplatform
from datetime import datetime

PROJECT_ID = "ml-coaching-team95-week3"  # @param {type:"string"}
REGION = "us-central1"  # @param {type: "string"}

BUCKET_NAME = "week3-training-124142000173"
PIPELINE_DEFINITION_BUCKET_NAME = "vertexai_pipeline_definition_ml-coaching-team95-week3"
EXPERIMENT_DESCRIPTION = "Running experiments in our first MLOps Pipeline"

PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipeline_root/mlops-coaching-part-one"


def process_request(request):
    """Processes the incoming HTTP request.

    Args:
      request (flask.Request): HTTP request object.

    Returns:
      The response text or any set of values that can be turned into a Response
      object using `make_response
      <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """

    TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
    EXPERIMENT_NAME = f"mlops-part-one-experimet-{TIMESTAMP}"
    # decode http request payload and translate into JSON object
    request_str = request.data.decode('utf-8')
    request_json = json.loads(request_str)

    parameter_values = request_json['parameter_values']

    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        experiment=EXPERIMENT_NAME,
        experiment_description=EXPERIMENT_DESCRIPTION
    )

    job=aiplatform.PipelineJob(
        display_name = "pipeline_run_with_data_validation",
        template_path = f"gs://{PIPELINE_DEFINITION_BUCKET_NAME}/pipeline.json",
        pipeline_root = PIPELINE_ROOT,
        enable_caching = True,
        parameter_values = parameter_values
    )

    job.submit(
        service_account="week3-training-sa@ml-coaching-team95-week3.iam.gserviceaccount.com"
    )
    return "Job submitted"
