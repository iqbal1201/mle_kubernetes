from kfp import dsl
from kfp.dsl import component
from kfp import compiler
from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs
import pandas as pd
import numpy as np
import pickle
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from google.oauth2 import service_account
import json
import os


aiplatform.init(project="ml-kubernetes-448516", location="us-central1")

# # Preprocessing component
@component(base_image="gcr.io/ml-kubernetes-448516/insurance-ml-app:latest")
def preprocess_data(input_csv: str, preprocessed_data_dir: str) -> str:
    import os
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    import pandas as pd
    import numpy as np
    import pickle

    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv(input_csv)
    print("Dataset loaded successfully.")

    # Separate features and target
    X = df.drop('charges', axis=1)
    y = df['charges']

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['age', 'bmi', 'children']),
            ('cat', OneHotEncoder(), ['sex', 'smoker', 'region'])
        ]
    )

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit and transform the data
    print("Preprocessing data...")
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    print("Data preprocessing completed.")

    # Save processed data
    os.makedirs(preprocessed_data_dir, exist_ok=True)
    np.save(os.path.join(preprocessed_data_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(preprocessed_data_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(preprocessed_data_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(preprocessed_data_dir, 'y_test.npy'), y_test)

    # Save preprocessor
    with open(os.path.join(preprocessed_data_dir, 'preprocessor.pkl'), 'wb') as f:
        pickle.dump(preprocessor, f)

    return preprocessed_data_dir


# @component(base_image="gcr.io/ml-kubernetes-448516/insurance-ml-app:latest")  # Explicitly specify the base image
# def preprocess_data(input_csv: str, preprocessed_data_dir: str) -> str:
#     import subprocess
#     import sys
#     import os
#     import pandas as pd
#     import numpy as np
#     import pickle
#     from sklearn.model_selection import train_test_split
#     from sklearn.preprocessing import StandardScaler, OneHotEncoder
#     from sklearn.compose import ColumnTransformer

#     try:
#         # Install dependencies
#         subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn==1.5.1", "pandas==2.2.2", "numpy"])

#         # Load dataset
#         print("Loading dataset...")
#         df = pd.read_csv(input_csv)
#         print("Dataset loaded successfully.")

#         # Separate features and target
#         X = df.drop('charges', axis=1)
#         y = df['charges']

#         # Preprocessing
#         preprocessor = ColumnTransformer(
#             transformers=[
#                 ('num', StandardScaler(), ['age', 'bmi', 'children']),
#                 ('cat', OneHotEncoder(), ['sex', 'smoker', 'region'])
#             ]
#         )

#         # Split the data
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         # Fit and transform the data
#         print("Preprocessing data...")
#         X_train = preprocessor.fit_transform(X_train)
#         X_test = preprocessor.transform(X_test)
#         print("Data preprocessing completed.")

#         # Save processed data
#         os.makedirs(preprocessed_data_dir, exist_ok=True)
#         np.save(os.path.join(preprocessed_data_dir, 'X_train.npy'), X_train)
#         np.save(os.path.join(preprocessed_data_dir, 'X_test.npy'), X_test)
#         np.save(os.path.join(preprocessed_data_dir, 'y_train.npy'), y_train)
#         np.save(os.path.join(preprocessed_data_dir, 'y_test.npy'), y_test)

#         # Save preprocessor
#         with open(os.path.join(preprocessed_data_dir, 'preprocessor.pkl'), 'wb') as f:
#             pickle.dump(preprocessor, f)

#         return preprocessed_data_dir

#     except Exception as e:
#         print(f"Error during preprocessing: {e}")
#         raise


# Model training component
@component(base_image="gcr.io/ml-kubernetes-448516/insurance-ml-app:latest")
def train_model(preprocessed_data_dir: str, model_output_dir: str) -> str:
    import os
    import numpy as np
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam

    # Load preprocessed data
    X_train = np.load(os.path.join(preprocessed_data_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(preprocessed_data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(preprocessed_data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(preprocessed_data_dir, 'y_test.npy'))

    # Define the model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])

    # Compile the model
    model.compile(
        loss='mae',
        optimizer=Adam(learning_rate=0.01),
        metrics=['mae']
    )

    # Train the model
    model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

    # Save the model
    os.makedirs(model_output_dir, exist_ok=True)
    model_path = os.path.join(model_output_dir, 'model_tf.h5')
    model.save(model_path)

    # Register the model in Vertex AI Model Registry in Vertex AI
    
    registered_model = aiplatform.Model.upload(
        display_name="tensorflow-regression-model",
        artifact_uri=model_output_dir,  # Path to the model directory in Cloud Storage
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest"
    )

    print(f"Model registered with ID: {registered_model.resource_name}")

    return model_output_dir



# Define the pipeline
@dsl.pipeline(
    name="tensorflow-regression-pipeline",
    pipeline_root="gs://ml-kubernetes-bucket/pipeline-root" 
)
def tensorflow_pipeline(
    input_csv: str,
    preprocessed_data_dir: str,
    model_output_dir: str
):
    preprocess_task = preprocess_data(
        input_csv=input_csv,
        preprocessed_data_dir=preprocessed_data_dir
    )

    train_model_task = train_model(
        preprocessed_data_dir=preprocess_task.output,
        model_output_dir=model_output_dir
    )


# Trigger the pipeline
if __name__ == "__main__":

    compiler.Compiler().compile(
        pipeline_func=tensorflow_pipeline,
        package_path="tensorflow_pipeline.json"
    )


    # Create and run the pipeline job
    job = pipeline_jobs.PipelineJob(
        display_name="tensorflow-regression-pipeline-job",
        template_path="tensorflow_pipeline.json",
        parameter_values={
            "input_csv": "gs://ml-kubernetes-bucket/insurance.csv",
            "preprocessed_data_dir": "gs://ml-kubernetes-bucket/preprocessed/",
            "model_output_dir": "gs://ml-kubernetes-bucket/models/"
        }
    )

    job.run()