"""Flask REST API for Autism Spectrum Disorder prediction."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

RAW_FEATURE_COLUMNS = [
    "age",
    "gender",
    "ethnicity",
    "jundice",
    "austim",
    "contry_of_res",
    "relation",
    "A1_Score",
    "A2_Score",
    "A3_Score",
    "A4_Score",
    "A5_Score",
    "A6_Score",
    "A7_Score",
    "A8_Score",
    "A9_Score",
    "A10_Score",
]

try:
    pipeline = joblib.load("autism_pipeline.pkl")
except FileNotFoundError:
    preprocessor = joblib.load("preprocessor.pkl")
    model = joblib.load("model.pkl")
    pipeline = None


def _validate_payload(payload: dict) -> pd.DataFrame:
    missing = [col for col in RAW_FEATURE_COLUMNS if col not in payload]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")
    
    try:
        age_val = float(payload["age"])
    except (TypeError, ValueError):
        raise ValueError("Age must be a valid number")
    
    if age_val < 18.0:
        raise ValueError("Prediction restricted to adults only (age must be >= 18)")
        
    row = {col: payload[col] for col in RAW_FEATURE_COLUMNS}
    return pd.DataFrame([row])


def _predict(input_df: pd.DataFrame) -> tuple[int, float]:
    if pipeline is not None:
        proba = pipeline.predict_proba(input_df)[0]
        pred = int(pipeline.predict(input_df)[0])
    else:
        features = preprocessor.transform(input_df)
        proba = model.predict_proba(features)[0]
        pred = int(model.predict(features)[0])
    return pred, float(proba[1])


@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "service": "Autism Detection API",
            "endpoints": {
                "GET /": "API information",
                "POST /predict": "Predict ASD status from feature JSON",
            },
            "required_fields": RAW_FEATURE_COLUMNS,
            "restrictions": {
                "age": "Prediction is restricted to adults only (age >= 18)"
            },
            "clinical_disclaimer": "This tool is for screening purposes only and does not replace professional medical diagnosis."
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    try:
        input_df = _validate_payload(payload)
        prediction, probability = _predict(input_df)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    label = "YES (ASD detected)" if prediction == 1 else "NO (ASD not detected)"
    return jsonify(
        {
            "prediction": label,
            "prediction_code": prediction,
            "probability": round(probability, 4),
            "probability_percent": f"{probability * 100:.2f}%",
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
