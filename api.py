from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
import os
import json
from shapely.geometry import shape, Point

# -----------------------------
# Load model + data
# -----------------------------
model = joblib.load("xgb_traffic_model.pkl")

if os.path.exists("traffic_training_df_lite.parquet"):
    df_model = pd.read_parquet("traffic_training_df_lite.parquet")

feature_cols = joblib.load("feature_cols.pkl")
valid_ntas = df_model["NTAName"].unique()

# -----------------------------
# Load lightweight NTA polygons
# -----------------------------
with open("nta_polygons.json") as f:
    NTA_POLYGONS = json.load(f)

def latlon_to_nta(lat, lon):
    """Returns the NTA name for a given (lat, lon) using shapely geometry."""
    point = Point(lon, lat)
    for rec in NTA_POLYGONS:
        polygon = shape(rec["geometry"])
        if polygon.contains(point):
            return rec["NTAName"]
    return None


app = FastAPI(
    title="NYC Traffic Speed Prediction API",
    description="Predict average traffic speed (mph) using an XGBoost model.",
    version="1.2.0"
)

# -----------------------------
# Request schema
# -----------------------------
class PredictRequest(BaseModel):
    lat: float
    lon: float
    timestamp: str
    num_events: int = 0
    num_closures: int = 0
    num_collisions: int = 0
    num_injured: int = 0
    avg_precip: float = 0.0


# -----------------------------
# Helper: compute lag features
# -----------------------------
def compute_lags(nta, ts):
    hist = df_model[df_model["NTAName"] == nta].sort_values("hour")
    past = hist[hist["hour"] < ts].tail(1)
    if past.empty:
        return None
    lag1 = past["avg_speed"].iloc[0]
    lag2 = past["lag1"].iloc[0]
    lag3 = past["lag2"].iloc[0]
    if pd.isna(lag1) or pd.isna(lag2) or pd.isna(lag3):
        return None
    return lag1, lag2, lag3


# -----------------------------
# Diagnostic endpoint
# -----------------------------
@app.get("/valid-ntas")
def list_ntas():
    return {"valid_ntas": sorted(valid_ntas.tolist())}


# -----------------------------
# MAIN /predict endpoint
# -----------------------------
@app.post("/predict")
def predict_speed(req: PredictRequest):

    # 1️⃣ Convert lat/lon → NTA
    nta = latlon_to_nta(req.lat, req.lon)
    if nta is None:
        return {"error": "Coordinates are outside NYC or not inside any NTA polygon."}

    # 2️⃣ Validate NTA exists in model data
    if nta not in valid_ntas:
        return {"error": f"NTA '{nta}' exists but was not part of training data."}

    # 3️⃣ Parse timestamp
    ts = pd.to_datetime(req.timestamp, errors="coerce")
    if pd.isna(ts):
        return {"error": "Invalid timestamp format"}

    # 4️⃣ Compute lag features
    lags = compute_lags(nta, ts)
    if lags is None:
        return {"error": "Insufficient history to compute lag features for this NTA."}

    lag1, lag2, lag3 = lags

    # 5️⃣ Encode NTA
    nta_code = df_model[df_model["NTAName"] == nta]["nta_code"].iloc[0]

    # 6️⃣ Build feature row
    X_new = pd.DataFrame([{
        "num_events": req.num_events,
        "num_closures": req.num_closures,
        "num_collisions": req.num_collisions,
        "num_injured": req.num_injured,
        "avg_precip": req.avg_precip,
        "hour_of_day": ts.hour,
        "day_of_week": ts.dayofweek,
        "nta_code": nta_code,
        "lag1": lag1,
        "lag2": lag2,
        "lag3": lag3
    }])

    # 7️⃣ Predict
    pred = float(model.predict(X_new)[0])

    return {
        "lat": req.lat,
        "lon": req.lon,
        "NTA_detected": nta,
        "timestamp": req.timestamp,
        "predicted_speed_mph": pred
    }
