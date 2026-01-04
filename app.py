import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained Ridge Regressor model and Standard Scaler
ridge_model = pickle.load(open('ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route("/", methods=['GET', 'POST'])
def home():
    prediction_result = None

    if request.method == 'POST':
        try:
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            Classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region', 0))

            new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
            result = ridge_model.predict(new_data_scaled)
            prediction_result = result[0]

        except Exception as e:
            print(f"Error: {e}")
            prediction_result = "Error"

    return render_template('index.html', prediction=prediction_result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
