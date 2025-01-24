from flask import Flask,request, url_for, redirect, render_template, jsonify
import tensorflow as tf
import pandas as pd
import pickle
import numpy as np
import config
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from keras.models import load_model
from google.cloud import storage



# GCS Bucket and Paths
GCS_BUCKET_NAME = "mle-kubernetes-bucket"  # Replace with your GCS bucket name
MODEL_PATH = "models/latest/model_tf.h5"  # Path to the model in GCS
PREPROCESSOR_PATH = "models/latest/preprocessor.pkl"  # Path to the preprocessor in GCS

# Local paths to store downloaded artifacts
LOCAL_MODEL_PATH = "model_tf.h5"
LOCAL_PREPROCESSOR_PATH = "preprocessor.pkl"


# Function to download files from GCS
def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """Download files from a GCS bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} to {destination_file_name}.")


# Download model and preprocessor when the app starts
if not os.path.exists(LOCAL_MODEL_PATH):
    download_from_gcs(GCS_BUCKET_NAME, MODEL_PATH, LOCAL_MODEL_PATH)

if not os.path.exists(LOCAL_PREPROCESSOR_PATH):
    download_from_gcs(GCS_BUCKET_NAME, PREPROCESSOR_PATH, LOCAL_PREPROCESSOR_PATH)

# Load model and preprocessor
model = tf.keras.models.load_model(LOCAL_MODEL_PATH)

with open(LOCAL_PREPROCESSOR_PATH, 'rb') as preprocessor_file:
    preprocessor = pickle.load(preprocessor_file)



app = Flask(__name__)

# model = load_model('deployment_28042020')
model = tf.keras.models.load_model('model_tf.h5')

with open('preprocessor.pkl', 'rb') as preprocessor_file:
    preprocessor = pickle.load(preprocessor_file)


cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    final = final.reshape(1, -1)
    data_unseen = pd.DataFrame(final, columns=cols)
    data_transformed = preprocessor.transform(data_unseen)
    # data_unseen = pd.DataFrame([final], columns = cols)
    prediction = model.predict(data_transformed)
    prediction = prediction[0]
    # prediction = int(prediction.Label[0])
    return render_template('home.html',pred='Expected Bill will be {}'.format(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    # data_unseen = pd.DataFrame([data])
    data_transformed = preprocessor.transform(data_unseen)
    # final = data.values
    # final = final.reshape(1, -1)
    prediction = model.predict(data_transformed)
    output = prediction[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG_MODE)
