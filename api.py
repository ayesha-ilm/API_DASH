from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
import os
import json
from rapidfuzz import process, fuzz


# -----------------------------
# Load model + data
# -----------------------------
model = joblib.load("xgb_traffic_model.pkl")

if os.path.exists("traffic_training_df_lite.parquet"):
    df_model = pd.read_parquet("traffic_training_df_lite.parquet")

feature_cols = joblib.load("feature_cols.pkl")
valid_ntas = df_model["NTAName"].unique().tolist()


# -----------------------------
# Load ALL official NYC NTAs (geojson → list)
# -----------------------------
with open("nta_polygons.json") as f:
    polygons_raw = json.load(f)

all_official_nta_names = [rec["NTAName"] for rec in polygons_raw]


app = FastAPI(
    title="NYC Traffic Speed Prediction API",
    description="Predict average traffic speed (mph) using an XGBoost model with fuzzy matching for NTAs.",
    version="1.3.0"
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
# Helper: fuzzy match to closest valid NTA
# -----------------------------
def fuzzy_match_nta(input_name):

    # If already valid, use as-is
    if input_name in valid_ntas:
        return input_name, 100.0

    # Try to match among ALL official NYC NTAs
    best_official, score_official, _ = process.extractOne(
        input_name,
        all_official_nta_names,
        scorer=fuzz.WRatio
    )

    # If the user typed something that isn't even close to ANY official NTA
    if score_official < 60:  
        return None, score_official

    # Best official → find nearest NTA among model-valid NTAs
    best_valid, score_valid, _ = process.extractOne(
        best_official,
        valid_ntas,
        scorer=fuzz.WRatio
    )

    return best_valid, score_valid


# -----------------------------
# Compute lag features
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
# Diagnostics
# -----------------------------
@app.get("/valid-ntas")
def list_ntas():
    return {"valid_ntas": sorted(valid_ntas), "count": len(valid_ntas)}


# -----------------------------
# MAIN /predict endpoint
# -----------------------------
@app.post("/predict")
def predict_speed(req: PredictRequest):

    # 1️⃣ Fuzzy match the NTA
    matched_nta, score = fuzzy_match_nta(req.NTAName)

    if matched_nta is None:
        return {
            "error": f"No reasonably close NTA found for '{req.NTAName}'",
            "match_score": score
        }

    # 2️⃣ Timestamp validation
    ts = pd.to_datetime(req.timestamp, errors="coerce")
    if pd.isna(ts):
        return {"error": "Invalid timestamp format"}

    # 3️⃣ Lags
    lags = compute_lags(matched_nta, ts)
    if lags is None:
        return {
            "error": "Insufficient history to compute lag features",
            "nta_used": matched_nta
        }

    lag1, lag2, lag3 = lags

    # 4️⃣ Encode NTA
    nta_code = df_model[df_model["NTAName"] == matched_nta]["nta_code"].iloc[0]

    # 5️⃣ Build feature row
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

    # 6️⃣ Predict
    pred = float(model.predict(X_new)[0])

    return {
        "input_NTA": req.NTAName,
        "matched_NTA": matched_nta,
        "match_confidence": score,
        "timestamp": req.timestamp,
        "predicted_speed_mph": pred
    }
