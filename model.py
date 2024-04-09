import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from datetime import datetime
import pickle
from sklearn.preprocessing import LabelEncoder
import sqlite3

db = sqlite3.connect("weather.db")


# Load the dataset
# data = pd.read_csv("weather.csv")
data = pd.read_sql_query("select * from weather", db)

# Load the dataset
data = pd.read_csv("weather.csv")


# Convert date strings to datetime objects
data["datetime"] = pd.to_datetime(data["datetime"], format="%d-%m-%Y")


# Extract day of the year as a feature (assuming the date follows some seasonality)
data["day_of_year"] = data["datetime"].dt.dayofyear


# Label encode the 'icon' column
label_encoder = LabelEncoder()
data["icon_encoded"] = label_encoder.fit_transform(data["icon"])


# Split the data into features and target variable
X = data[["day_of_year"]]

y_precip = data["precip"]
y_precipCover = data["precipcover"]
y_humidity = data["humidity"]
y_windspeed = data["windspeed"]
y_cloudCover = data["cloudcover"]
y_icon = data["icon_encoded"]
y_tempmax=data["tempmax"]
y_tempmin=data["tempmin"]

model_tempmax = RandomForestRegressor(n_estimators=100, random_state=42)
model_tempmax.fit(X, y_tempmax)

model_tempmin = RandomForestRegressor(n_estimators=100, random_state=42)
model_tempmin.fit(X, y_tempmin)

model_precip = RandomForestRegressor(n_estimators=100, random_state=42)
model_precip.fit(X, y_precip)

model_precipCover = RandomForestRegressor(n_estimators=100, random_state=42)
model_precipCover.fit(X, y_precipCover)

model_humidity = RandomForestRegressor(n_estimators=100, random_state=42)
model_humidity.fit(X, y_humidity)

model_windspeed = RandomForestRegressor(n_estimators=100, random_state=42)
model_windspeed.fit(X, y_windspeed)

model_cloudCover = RandomForestRegressor(n_estimators=100, random_state=42)
model_cloudCover.fit(X, y_cloudCover)

model_icon = RandomForestClassifier(n_estimators=100, random_state=42)
model_icon.fit(X, y_icon)

# Dump the dictionary containing all models into a pickle file
with open("model_tempmax.pkl", "wb") as f:
    pickle.dump(model_tempmax, f)

with open("model_tempmin.pkl", "wb") as f:
    pickle.dump(model_tempmin, f)

with open("model_precip.pkl", "wb") as f:
    pickle.dump(model_precip, f)

with open("model_precipCover.pkl", "wb") as f:
    pickle.dump(model_precipCover, f)

with open("model_humidity.pkl", "wb") as f:
    pickle.dump(model_humidity, f)

with open("model_windspeed.pkl", "wb") as f:
    pickle.dump(model_windspeed, f)

with open("model_cloudCover.pkl", "wb") as f:
    pickle.dump(model_cloudCover, f)

with open("model_icon.pkl", "wb") as f:
    pickle.dump(model_icon, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)


# loading
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
