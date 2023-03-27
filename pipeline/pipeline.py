import os
import shutil
import logging
from datetime import datetime
#import pandas as pd
import numpy as np
logger = logging.getLogger("logger")
logging.basicConfig(level=logging.INFO)

# Import Google Cloud libraries
from google.cloud import aiplatform as aip
from google.cloud.aiplatform import pipeline_jobs
import google.auth
from google.cloud import storage
from google_cloud_pipeline_components import aiplatform as gcc_aip
from google.cloud import bigquery
from google_cloud_pipeline_components.types import artifact_types
from google_cloud_pipeline_components.v1.model import ModelUploadOp
from google_cloud_pipeline_components.v1.endpoint import EndpointCreateOp, ModelDeployOp


# Import kfp sdk modules to create the Vertex AI Pipelines
from kfp.v2 import compiler, dsl
from kfp.v2.dsl import pipeline

from typing import NamedTuple

from kfp.v2.dsl import (Artifact, Dataset, Input, Model, Output, Metrics, ClassificationMetrics, HTML, component, OutputPath, InputPath)
from kfp.v2 import compiler

from kfp.v2.dsl import importer
import kfp.v2.compiler as compiler


# Vertex AI
from google.cloud import aiplatform
from google.cloud.aiplatform_v1.types.pipeline_state import PipelineState
from kfp.v2.dsl import Metrics, Model, Output, component, Input, Dataset, Condition, Artifact, HTML
from kfp.v2.components import importer_node

# TensorFlow model building libraries.
# import tensorflow as tf

PROJECT_ID = "ml-coaching-team95-week3"  # @param {type:"string"}
REGION = "us-central1"  # @param {type: "string"}
    
BUCKET_NAME = f"week3-training-124142000173"
BUCKET_URI = f"gs://week3-training-124142000173"

TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")

EXPERIMENT_NAME = f"mlops-part-one-experimet-{TIMESTAMP}"
EXPERIMENT_DESCRIPTION = "Running experiments in our first MLOps Pipeline"
PIPELINE_NAME = f"mlops-part-one-pipeline-{TIMESTAMP}"
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipeline_root/mlops-coaching-part-one"
PIPELINE_DISPLAY_NAME = "mlops-part-one-pipeline"
COMPILED_PIPELINE_PACKAGE_PATH = "mlops-part-one-pipeline.json"

aip.init(project=PROJECT_ID, 
         staging_bucket=BUCKET_NAME,
         experiment=EXPERIMENT_NAME,
         experiment_description=EXPERIMENT_DESCRIPTION)

@component(
    packages_to_install=["pandas", "google-cloud-bigquery[bqstorage,pandas]"],
    output_component_file="split_dataset_op.yml" 
)
def split_dataset_op(
    bq_table_uri: str,
    dataset_train: Output[Dataset],
    dataset_test: Output[Dataset],
    split_frac: float =0.8, 
    random_state: int = 0
): 
    from google.cloud import bigquery
    import pandas as pd
    import logging
    import os
    
    bqclient = bigquery.Client()
    
    def download_table(bq_table_uri: str):
        prefix = "bq://"
        if bq_table_uri.startswith(prefix):
            bq_table_uri = bq_table_uri[len(prefix):]

        table = bigquery.TableReference.from_string(bq_table_uri)
        rows = bqclient.list_rows(
            table,
        )
        return rows.to_dataframe(create_bqstorage_client=False)
    
    
    raw_dataset = download_table(bq_table_uri)
    raw_dataset.rename(columns = {
        'y':'target'}, inplace = True)

    # Get data in shape
    dataset = raw_dataset.copy()
    dataset = dataset.dropna()
    dataset['target'] = dataset['target'].map({'yes':1, 'no':0})
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    dataset_train.metadata['split method']= 'random sampling, 80% of data used for training'
    dataset_test.metadata['split method']= 'random sampling, 20% of data used for validation'
    train_dataset.to_csv(dataset_train.path, index=False)
    test_dataset.to_csv(dataset_test.path, index=False)
    
    logging.info(f"Dataset Train is be stored in: {dataset_train.path}")
    logging.info(f"Dataset Test is be stored in: {dataset_test.path}")
    
@component(
    packages_to_install=[
        "pandas",
        "numpy",
        "tensorflow==2.8.0"
    ],
    output_component_file="train_model_op.yml"
)
def custom_trainer_op(
    train_dataset: Input[Dataset],
    batch_size: int,
    num_units: int,
    epochs: int,
    dropout_rate:float,
    model_artifact: Output[Model],
    metrics_metadata: Output[Metrics]
)-> str:
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from tensorflow.keras import layers
    import logging
    
    
    def df_to_dataset(dataframe, target_col, shuffle=True, batch_size=32):
        '''
        Transform a Pandas Dataframe to a tf.data.Dataset. This helps to parallelize and
        optimize input pipelines.
        '''
        df = dataframe.copy()
        labels = df.pop(target_col)
        features = df
        ds = tf.data.Dataset.from_tensor_slices((dict(features), labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))
            ds = ds.batch(batch_size)
            ds = ds.prefetch(batch_size)
        return ds
    
    def get_normalization_layer(col_name, dataset):
        '''
        Function to normalize numerical variables
        ''' 
        # Create a Normalization layer for the feature.
        normalizer = layers.Normalization(axis=None)

        # Prepare a Dataset that only yields the feature.
        feature_ds = dataset.map(lambda x, y: x[col_name])

        # Learn the statistics of the data.
        normalizer.adapt(feature_ds)
        return normalizer

    def get_category_encoding_layer(col_name, dataset, dtype, max_tokens=None):
        '''
        Function to transform categorical variables in order to be handled by our NN.
        '''
        # Create a layer that turns strings into integer indices.
        if dtype == 'string':
            index = layers.StringLookup(max_tokens=max_tokens)

        # Otherwise, create a layer that turns integer values into integer indices.
        else:
            index = layers.IntegerLookup(max_tokens=max_tokens)

        # Prepare a `tf.data.Dataset` that only yields the feature.
        feature_ds = dataset.map(lambda x, y: x[col_name])

        # Learn the set of possible values and assign them a fixed integer index.
        index.adapt(feature_ds)
        # Encode the integer indices.
        encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size())
        # Apply multi-hot encoding to the indices. The lambda function captures the
        # layer, so you can use them, or include them in the Keras Functional model later.
        return lambda feature: encoder(index(feature))
    
    
    # Read Data from output dataset from the first step.
    df_train = pd.read_csv(train_dataset.path)
    
    
    # Lists of numerical and categorical features
    NUM_FEATS = ["age", "balance", "duration", "pdays", "previous"]
    CAT_FEATS = ["job", "marital", "education", "default", "housing", "loan", "contact", "day", "month", "campaign", "poutcome"]
    target_col = "target"
    
    # Ensure Numerical Features are of type float and Categorical of type object
    df_train[NUM_FEATS] =df_train[NUM_FEATS].astype(np.float32)
    df_train[CAT_FEATS] =df_train[CAT_FEATS].astype(str)
    
    # Converte pandas dataframe to tf.Data.Dataset

    train_ds = df_to_dataset(dataframe=df_train, 
                             target_col=target_col, 
                             shuffle=True, 
                             batch_size=batch_size)

    # Normalize and Encode numerical and categorical features, respectively
    all_inputs = []
    encoded_features = []

    # Normalize all numerical features
    for header in NUM_FEATS:
        numeric_col = tf.keras.Input(shape=(1,), name=header)
        normalization_layer = get_normalization_layer(header, train_ds)
        encoded_numeric_col = normalization_layer(numeric_col)
        all_inputs.append(numeric_col)
        encoded_features.append(encoded_numeric_col)


    # One-hot-encode all categorical features
    for header in CAT_FEATS:
        categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
        encoding_layer = get_category_encoding_layer(col_name=header,
                                               dataset=train_ds,
                                               dtype='string',
                                               max_tokens=5)
        encoded_categorical_col = encoding_layer(categorical_col)
        all_inputs.append(categorical_col)
        encoded_features.append(encoded_categorical_col)

        
    # Define The Model including on it the preprocessing layer
        
    all_features = tf.keras.layers.concatenate(encoded_features)
    x = tf.keras.layers.Dense(num_units, activation="relu")(all_features)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    output = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(all_inputs, output)
    
    # Compile The Model
    model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=["accuracy"])

    model.fit(train_ds, epochs=epochs)
    
    # Save the model in our output artifact uri
    model.save(model_artifact.uri)
    
    ## Evaluate and log metrics on Train Dataset
    loss, accuracy = model.evaluate(train_ds)

    # Log the metrics for comparison and lineage purposes
    metrics_metadata.log_metric("train_loss", loss)
    metrics_metadata.log_metric("train_accuracy", accuracy)
    
    logging.info(f"Model assets are stored on: {model_artifact.uri}")
    return model_artifact.uri

    
    
  
@component(
    packages_to_install=[
        "pandas",
        "gcsfs",
        "tensorflow==2.8.0"
    ]
)
def evaluate_op(
    test_dataset: Input[Dataset],
    batch_size: int,
    model: Input[Model],
    metrics_metadata: Output[Metrics],
)-> float:
    
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import logging
    
    def df_to_dataset(dataframe, target_col, shuffle=True, batch_size=32):
        '''
        Transform a Pandas Dataframe to a tf.data.Dataset. This helps to parallelize and
        optimize input pipelines.
        '''
        df = dataframe.copy()
        labels = df.pop(target_col)
        features = df
        ds = tf.data.Dataset.from_tensor_slices((dict(features), labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))
            ds = ds.batch(batch_size)
            ds = ds.prefetch(batch_size)
        return ds
    
    # Read Dataset from output dataset from the first step. 
    df_test = pd.read_csv(test_dataset.path)
    # Lists of numerical and categorical features
    NUM_FEATS = ["age", "balance", "duration", "pdays", "previous"]
    CAT_FEATS = ["job", "marital", "education", "default", "housing", "loan", "contact", "day", "month", "campaign", "poutcome"]
    target_col = "target"
    
    
    # Ensure Numerical Features are of type float and Categorical of type object
    df_test[NUM_FEATS] =df_test[NUM_FEATS].astype(np.float32)
    df_test[CAT_FEATS] =df_test[CAT_FEATS].astype(str)
    
    # Transform our pandas dataframe to a tf.data.Dataset
    test_ds = df_to_dataset(df_test, target_col, shuffle=True, batch_size=32)
    
    #Load our model
    model_object = tf.keras.models.load_model(model.path)
    
    
    # Evaluate our model using our test dataset
    loss, accuracy = model_object.evaluate(test_ds)

    # Log the metrics for comparison and lineage purposes
    metrics_metadata.log_metric("test_loss", loss)
    metrics_metadata.log_metric("test_accuracy", accuracy)
    return accuracy

@component(
    packages_to_install=[
        "tensorflow-data-validation"
    ]
)
def generate_statistics_op(
    train_dataset: Input[Dataset],
    test_dataset: Input[Dataset],
    train_statistics: Output[Artifact],
    train_statistics_view: Output[HTML],
    test_statistics: Output[Artifact],
    test_statistics_view: Output[HTML]
):
    import tensorflow_data_validation as tfdv
    from tensorflow_data_validation.utils.display_util import get_statistics_html

    train_dataset_statistics =  tfdv.generate_statistics_from_csv(
        data_location=train_dataset.uri, output_path=train_statistics.uri
    )

    html_content = get_statistics_html(lhs_statistics=train_dataset_statistics)
    train_statistics_view.path = f"{train_statistics_view.path}"
    with open(train_statistics_view.path, "w") as f:
        f.write(html_content)
        
    
    test_dataset_statistics =  tfdv.generate_statistics_from_csv(
        data_location=test_dataset.uri, output_path=test_statistics.uri
    )

    html_content = get_statistics_html(lhs_statistics=test_dataset_statistics)
    test_statistics_view.path = f"{test_statistics_view.path}"
    with open(test_statistics_view.path, "w") as f:
        f.write(html_content)

@component(
    packages_to_install=[
        "tensorflow-data-validation"
    ]
)
def generate_statistics_view_comparison_op(
    train_statistics: Input[Artifact],
    test_statistics: Input[Artifact],
    statistics_view: Output[HTML],
    lhs_name: str = "lhs_statistics",
    rhs_name: str = "rhs_statistics"
):
    import tensorflow_data_validation as tfdv
    from tensorflow_data_validation.utils.display_util import get_statistics_html

    train_stats = tfdv.load_statistics(input_path=train_statistics.uri)
    test_stats = tfdv.load_statistics(input_path=test_statistics.uri)
    html_content = get_statistics_html(
        lhs_statistics=train_stats,
        rhs_statistics=test_stats,
        lhs_name=lhs_name,
        rhs_name=rhs_name,
    )

    with open(statistics_view.path, "w") as f:
        f.write(html_content)

        
@component(
    packages_to_install=[
        "tensorflow-data-validation"
    ]
)
def validate_test_data_op(
    train_statistics: Input[Artifact],
    test_statistics: Input[Artifact],
    schema_inferred: Output[Artifact],
    anomalies_detected: Output[Artifact]
)-> str:
    
    import tensorflow_data_validation as tfdv
    from tensorflow_data_validation.utils.display_util import get_statistics_html
    from tensorflow_data_validation import write_anomalies_text, write_schema_text
    
    train_stats = tfdv.load_statistics(input_path=train_statistics.uri)
    test_stats = tfdv.load_statistics(input_path=test_statistics.uri)
    
    schema = tfdv.infer_schema(train_stats)
    tfdv.display_schema(schema)
    
    anomalies = tfdv.validate_statistics(statistics=test_stats, schema=schema)
    
    
    
    if anomalies.anomaly_info == dict():
        anomalies_detected_th = "False"
        anomalies_detected.metadata['result'] =  "No anomalies found"
    else:
        anomalies_detected_th = "True"
        anomalies_detected.metadata['result'] = "Anomalies found"
        
    
    write_anomalies_text(anomalies, anomalies_detected.path)
    write_schema_text(schema, schema_inferred.path)
    
    return anomalies_detected_th
    
@dsl.pipeline(name="mlops-custom-training-pipeline")
def pipeline(
    project: str = PROJECT_ID,
    region: str = REGION,
    bq_table_uri: str="", 
    split_frac: float=0.8,
    random_state: int=0,
    epochs: int = 2,
    dropout_rate: float = 0.5,
    batch_size: int= 32,
    num_units: int = 32,
    model_accuracy_th: float = 0.85,
    deploy_model: str = "False",
    endpoint_name: str = "term-deposit-predictor"
):
    # Extract Data from BigQuery and split it into Train and Test Datasets
    extract_and_split_data_task = split_dataset_op(
        bq_table_uri=bq_table_uri,
        split_frac=split_frac,
        random_state=random_state
    ).set_cpu_limit('4').set_memory_limit('16G')   
    
    ###  Data Validation
    
    get_statistics_task = generate_statistics_op(
        train_dataset=extract_and_split_data_task.outputs['dataset_train'],
        test_dataset=extract_and_split_data_task.outputs['dataset_test']
    ) # TODO - Complete with the outputs of previous tasks
    
    _ = generate_statistics_view_comparison_op(
        train_statistics=get_statistics_task.outputs['train_statistics'],
        test_statistics=get_statistics_task.outputs['test_statistics']
    ) # TODO - Complete with the outputs of previous tasks
    
    anomalies_detected = validate_test_data_op(
      train_statistics=get_statistics_task.outputs['train_statistics'],
      test_statistics=get_statistics_task.outputs['test_statistics']  
    ).outputs['Output'] #TODO # TODO - Complete with the outputs of previous tasks. 
    
    with dsl.Condition(
        anomalies_detected != "True",
        name="no_anomalies"
    ):
        # Custom Training 
        custom_training_task = custom_trainer_op(
            train_dataset=extract_and_split_data_task.outputs['dataset_train'],
            batch_size=batch_size,
            num_units=num_units,
            epochs=epochs,
            dropout_rate=dropout_rate,
        ).set_cpu_limit('4').set_memory_limit('16G')

    
        # Model Evaluation
        model_accuracy = evaluate_op(
            test_dataset=extract_and_split_data_task.outputs['dataset_test'],
            batch_size=batch_size,
            model=custom_training_task.outputs['model_artifact'],
        ).set_cpu_limit('4').set_memory_limit('16G').outputs['Output'] 
        
        
        ### @TODO Add a condition in such a way that if the model_accuracy is larger than the model accuracy threshold specified it will go
        # throught he next steps
        
        with dsl.Condition(
            model_accuracy > model_accuracy_th,
            name="accuracy_accepted"
        ):
            with dsl.Condition(
                deploy_model == "True",
                name="deploy_enabled ",
            ):
                managed_model = importer_node.importer(
                    artifact_uri=custom_training_task.outputs['output'],
                    artifact_class=artifact_types.UnmanagedContainerModel,
                    metadata={
                        "containerSpec": {
                            "imageUri": "us-docker.pkg.dev/cloud-aiplatform/prediction/tf2-cpu.2-9:latest"
                        }
                }).outputs["artifact"]

                model_upload_op = ModelUploadOp(
                    project=PROJECT_ID,
                    location=REGION,
                    display_name="term_deposit_prediction",
                    unmanaged_container_model=managed_model
                ) # TODO - Complete with the params required and outputs of previous tasks

                endpoint_op = EndpointCreateOp(
                    project=PROJECT_ID,
                    location=REGION,
                    display_name=endpoint_name,
                ) # TODO - Complete with the params required and outputs of previous tasks

                _ = ModelDeployOp(
                    model=model_upload_op.outputs["model"],
                    endpoint=endpoint_op.outputs["endpoint"],
                    dedicated_resources_min_replica_count=1,
                    dedicated_resources_max_replica_count=1,
                    dedicated_resources_machine_type="n1-standard-4",
                    service_account="week3-training-sa@ml-coaching-team95-week3.iam.gserviceaccount.com"      
                )  # TODO - Complete with the params required and outputs of previous tasks                 

compiler.Compiler().compile(
    pipeline_func=pipeline,
    package_path='pipeline.json',
) #Todo, add pipeline_func and package path params