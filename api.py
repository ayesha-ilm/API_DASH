from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
import os
# -----------------------------
# Load model + data
# -----------------------------
model = joblib.load("xgb_traffic_model.pkl")
if os.path.exists("traffic_training_df_lite.parquet"):
    df_model = pd.read_parquet("traffic_training_df_lite.parquet")
feature_cols = joblib.load("feature_cols.pkl")
valid_ntas = df_model["NTAName"].unique()

app = FastAPI(
    title="NYC Traffic Speed Prediction API",
    description="Predict average traffic speed (mph) using an XGBoost model.",
    version="1.0.0"
)

# -----------------------------
# Request schema
# -----------------------------
class PredictRequest(BaseModel):
    NTAName: str
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
# MAIN ENDPOINT
# -----------------------------
@app.post("/predict")
def predict_speed(req: PredictRequest):

    # Validate NTA
    if req.NTAName not in valid_ntas:
        return {"error": f"Invalid NTAName '{req.NTAName}'"}

    # Validate timestamp
    ts = pd.to_datetime(req.timestamp, errors="coerce")
    if pd.isna(ts):
        return {"error": "Invalid timestamp format"}

    # Compute lags
    lags = compute_lags(req.NTAName, ts)
    if lags is None:
        return {"error": "Insufficient history to compute lag features"}

    lag1, lag2, lag3 = lags

    # Encode NTA
    nta_code = df_model[df_model["NTAName"] == req.NTAName]["nta_code"].iloc[0]

    # Build feature row
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

    # Prediction
    pred = float(model.predict(X_new)[0])

    return {
        "nta": req.NTAName,
        "timestamp": req.timestamp,
        "predicted_speed_mph": pred
    }
