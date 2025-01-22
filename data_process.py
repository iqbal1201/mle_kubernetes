import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import numpy as np
import os

data_path = os.path.join(os.path.dirname(__file__), 'src', 'insurance.csv')
df = pd.read_csv(data_path, delimiter=',')



# Separate features and target
X = df.drop('charges', axis=1)
y = df['charges']


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['age', 'bmi', 'children']),
        ('cat', OneHotEncoder(), ['sex', 'smoker', 'region'])
    ])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit and transform the data
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

np.save(r'C:\Iqbal\Project\ml_kubernetes\src\X_train.npy', X_train)
np.save(r'C:\Iqbal\Project\ml_kubernetes\src\X_test.npy', X_test)
np.save(r'C:\Iqbal\Project\ml_kubernetes\src\y_train.npy', y_train)
np.save(r'C:\Iqbal\Project\ml_kubernetes\src\y_test.npy', y_test)

with open('preprocessor.pkl', 'wb') as preprocessor_file:
    pickle.dump(preprocessor, preprocessor_file)