from flask import Flask,request, url_for, redirect, render_template, jsonify
import tensorflow as tf
import pandas as pd
import pickle
import numpy as np
import config
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from keras.models import load_model

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
