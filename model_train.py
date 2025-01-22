import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


X_train = np.load(r'C:\Iqbal\Project\ml_kubernetes\src\X_train.npy')
X_test = np.load(r'C:\Iqbal\Project\ml_kubernetes\src\X_test.npy')
y_train = np.load(r'C:\Iqbal\Project\ml_kubernetes\src\y_train.npy')
y_test = np.load(r'C:\Iqbal\Project\ml_kubernetes\src\y_test.npy')


model = keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='linear')
])

model.compile(loss = tf.keras.losses.mean_absolute_error,
               optimizer = tf.keras.optimizers.Adam(lr=0.01),
               metrics=['mae'])



history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test,y_test))

model.save('model_tf.h5')