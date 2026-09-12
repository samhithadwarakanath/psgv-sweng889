import os

import joblib
import pandas as pd

FEATURE_ORDER = [
    "Hour",
    "Temperature(°C)",
    "Humidity(%)",
    "Wind speed (m/s)",
    "Visibility (10m)",
    "Dew point temperature(°C)",
    "Solar Radiation (MJ/m2)",
    "Rainfall(mm)",
    "Snowfall (cm)",
    "Holiday",
    "Functioning Day",
    "Seasons_Spring",
    "Seasons_Summer",
    "Seasons_Winter",
    "month",
    "day_of_week",
    "is_weekend",
    "is_peak_hour",
    "is_night",
]

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "bike_demand_model.joblib",
)

MIN_TEMPERATURE = -20.0
MAX_TEMPERATURE = 40.0

DEFAULT_HUMIDITY = 60
DEFAULT_WIND_SPEED = 2.0
DEFAULT_VISIBILITY = 1500
DEFAULT_SOLAR_RADIATION = 0.5
DEFAULT_RAINFALL = 0
DEFAULT_SNOWFALL = 0

PEAK_HOURS = {7, 8, 9, 18, 19, 20}
NIGHT_HOURS = {22, 23, 0, 1, 2, 3, 4, 5}

_model = None


def _load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def build_features(date, time, temperature, is_holiday) -> pd.DataFrame:
    if date is None or time is None:
        raise ValueError("Date and time are required.")
    try:
        temperature = float(temperature)
    except (TypeError, ValueError):
        raise ValueError("Temperature must be a number.")
    if not (MIN_TEMPERATURE <= temperature <= MAX_TEMPERATURE):
        raise ValueError(
            f"Temperature must be between {MIN_TEMPERATURE}°C and {MAX_TEMPERATURE}°C."
        )

    hour = time.hour
    day_of_week = date.weekday()
    month = date.month

    row = {
        "Hour": hour,
        "Temperature(°C)": temperature,
        "Humidity(%)": DEFAULT_HUMIDITY,
        "Wind speed (m/s)": DEFAULT_WIND_SPEED,
        "Visibility (10m)": DEFAULT_VISIBILITY,
        "Dew point temperature(°C)": temperature - 5,
        "Solar Radiation (MJ/m2)": DEFAULT_SOLAR_RADIATION,
        "Rainfall(mm)": DEFAULT_RAINFALL,
        "Snowfall (cm)": DEFAULT_SNOWFALL,
        "Holiday": 1 if is_holiday else 0,
        "Functioning Day": 1,
        "Seasons_Spring": 1 if month in (3, 4, 5) else 0,
        "Seasons_Summer": 1 if month in (6, 7, 8) else 0,
        "Seasons_Winter": 1 if month in (12, 1, 2) else 0,
        "month": month,
        "day_of_week": day_of_week,
        "is_weekend": 1 if day_of_week >= 5 else 0,
        "is_peak_hour": 1 if hour in PEAK_HOURS else 0,
        "is_night": 1 if hour in NIGHT_HOURS else 0,
    }

    return pd.DataFrame([row], columns=FEATURE_ORDER)


def predict_demand(date, time, temperature, is_holiday) -> int:
    features = build_features(date, time, temperature, is_holiday)
    model = _load_model()
    prediction = model.predict(features)[0]
    return int(max(0, round(prediction)))
