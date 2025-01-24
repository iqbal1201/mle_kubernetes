from kfp.v2 import dsl
from kfp.v2.dsl import component
from google.cloud import aiplatform
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Preprocessing component
@component
def preprocess_data(input_csv: str, preprocessed_data_dir: str) -> str:
    import os
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    import pandas as pd
    import numpy as np
    import pickle

    # Load dataset
    df = pd.read_csv(input_csv)

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
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

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


# Model training component
@component
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
if __name__ == "_main_":
    aiplatform.init(project="ml-kubernetes-448516", location="us-central1")

    # Create and run the pipeline job
    job = aiplatform.PipelineJob(
        display_name="tensorflow-regression-pipeline-job",
        template_path="tensorflow_pipeline.json",
        parameter_values={
            "input_csv": "gs://ml-kubernetes-bucket/insurance.csv",
            "preprocessed_data_dir": "gs://ml-kubernetes-bucket/preprocessed/",
            "model_output_dir": "gs://ml-kubernetes/models-bucket/"
        }
    )

    job.run()