import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
import pickle
import os
import joblib

os.chdir(os.path.dirname(__file__))

data = pd.read_csv('data/winequality.csv', sep=";")

X = data[['alcohol', 'pH', 'sulphates']]
y = data['quality'] 

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.20,
                                                    random_state=42)

model = XGBRegressor(max_depth=10, learning_rate=0.1, n_estimators=1000, random_state=42)


model.fit(X, y) #entrenamos con todos los datos

with open('ad_model.pkl', 'wb') as f:
    pickle.dump(model, f)
