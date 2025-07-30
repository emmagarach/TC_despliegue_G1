from flask import Flask, jsonify, request, render_template
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np

# os.chdir(os.path.dirname(__file__))

app = Flask(__name__)


# Landing page
@app.route('/', methods=['GET'])
def hello():
    return render_template('index.html')

# Predict
@app.route('/api/v1/predict', methods=['GET'])
def predict():
    with open('ad_model.pkl', 'rb') as f:
        model = pickle.load(f)

    alcohol = request.args.get('alcohol', None)
    ph = request.args.get('pH', None)
    sulphates = request.args.get('sulphates', None)

    if alcohol is None or ph is None or sulphates is None:
        return "Args empty, not enough data to predict", 400  # Mejor devolver un código HTTP de error

    try:
        features = [float(alcohol), float(ph), float(sulphates)]
    except ValueError:
        return "Invalid input: unable to convert to float", 400

    prediction = model.predict([features])

    return jsonify({'prediction': round(float(prediction[0]),2)})

#Retrain
@app.route("/api/v1/retrain", methods=["GET"])
def retrain():
    if os.path.exists("data/winequality_new.csv"):
        data = pd.read_csv('data/winequality_new.csv', sep=";")

        X = data[['alcohol', 'pH', 'sulphates']]
        y = data['quality'] 

        X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.20,
                                                    random_state=42)

        model = XGBRegressor(max_depth=10, learning_rate=0.1, n_estimators=1000, random_state=42)
        model.fit(X_train, y_train)
        rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        mape = mean_absolute_percentage_error(y_test, model.predict(X_test))
        model.fit(X,y)
        with open('ad_model.pkl', 'wb') as f:
            pickle.dump(model, f)
            
        return f"Model retrained. New evaluation metric RMSE: {str(rmse)}, MAPE: {str(mape)}"
    else:
        return f"<h2>New data for retrain NOT FOUND. Nothing done!</h2>"

#Redeploy
@app.route("/ping", methods=["GET"])
def ping():
    return {"mensaje": "API funcionando correctamente. Versión 1.0"}

if __name__ == '__main__':
    app.run(debug=True)
