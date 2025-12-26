from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("ridge.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        Temperature = float(request.form["Temperature"])
        RH = float(request.form["RH"])
        Ws = float(request.form["Ws"])
        Rain = float(request.form["Rain"])
        FFMC = float(request.form["FFMC"])
        DMC = float(request.form["DMC"])
        DC = float(request.form["DC"])
        ISI = float(request.form["ISI"])
        BUI = float(request.form["BUI"])

        features = np.array([[
            Temperature, RH, Ws, Rain,
            FFMC, DMC, DC, ISI, BUI
        ]])

        scaled_features = scaler.transform(features)
        prediction = round(model.predict(scaled_features)[0], 2)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)

