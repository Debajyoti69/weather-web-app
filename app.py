from flask import Flask, request, render_template
import pickle
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)


# Load all your models
with open("model_tempmax.pkl", "rb") as f:
    model_tempmax = pickle.load(f)

with open("model_tempmin.pkl", "rb") as f:
    model_tempmin = pickle.load(f)

with open("model_precip.pkl", "rb") as f:
    model_precip = pickle.load(f)

with open("model_precipCover.pkl", "rb") as f:
    model_precipCover = pickle.load(f)

with open("model_humidity.pkl", "rb") as f:
    model_humidity = pickle.load(f)

with open("model_windspeed.pkl", "rb") as f:
    model_windspeed = pickle.load(f)

with open("model_cloudCover.pkl", "rb") as f:
    model_cloudCover = pickle.load(f)

with open("model_icon.pkl", "rb") as f:
    model_icon = pickle.load(f)

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)


@app.route("/")
def hello_world():
    # return render_template("index.html")
    return render_template("index.html")


@app.route("/predict", methods=["POST", "GET"])
def predict():
    # try:
    day = request.form["day"]
    month = request.form["month"]
    year = request.form["year"]

    # Combine day, month, and year into a single string in dd-mm-yyyy format
    date_string = f"{day}-{month}-{year}"
    # day_of_year = date_string.timetuple().tm_yday
    day_of_year = datetime.strptime(date_string, "%d-%m-%Y").timetuple().tm_yday
    day_of_year = np.array(day_of_year).reshape(1, -1)

    tempmax_prediction=model_tempmax.predict(day_of_year)[0]
    tempmin_prediction=model_tempmin.predict(day_of_year)[0]

    precip_prediction = model_precip.predict(day_of_year)[0]
    precipCover_prediction = model_precipCover.predict(day_of_year)[0]
    humidity_prediction = model_humidity.predict(day_of_year)[0]
    windspeed_prediction = model_windspeed.predict(day_of_year)[0]
    cloudCover_prediction = model_cloudCover.predict(day_of_year)[0]

    # # Prediction for the icon
    # icon_prediction_encoded = model_icon.predict(day_of_year)[0]
    # icon_prediction = model_icon.inverse_transform([icon_prediction_encoded])[0]
    # Predict icon label

    icon_prediction_encoded = model_icon.predict(day_of_year)[0]
    # Convert encoded label back to original label
    icon_prediction = label_encoder.inverse_transform([icon_prediction_encoded])[0]
    # icon_prediction = model_icon.inverse_transform(model_icon.predict(day_of_year))[0]

    return render_template(
        "index.html",
        tempmax_prediction=round(tempmax_prediction,2),
        tempmin_prediction=round(tempmin_prediction,2),
        precip_prediction=round(precip_prediction, 2),
        precipCover_prediction=round(precipCover_prediction, 2),
        humidity_prediction=round(humidity_prediction, 2),
        windspeed_prediction=round(windspeed_prediction, 2),
        cloudCover_prediction=round(cloudCover_prediction, 2),
        icon_prediction=icon_prediction,
    )


if __name__ == "__main__":
    app.run(debug=True)
