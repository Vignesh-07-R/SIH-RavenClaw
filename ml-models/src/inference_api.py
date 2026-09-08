"""
Minimal API packaging anomaly + RUL + fault classification into one
health_report (see docs/DATA_CONTRACT.md), for dashboard to consume.

Requires anomaly_detector.py, rul_model.py, and fault_classifier.py to have
been run once already (`python src/<name>.py`) so their .joblib files exist.

Run with: python src/inference_api.py
Then: curl http://localhost:8002/health/latest
"""

from datetime import datetime, timezone

import requests
from fastapi import FastAPI
import uvicorn

import anomaly_detector
import rul_model
import fault_classifier
from features import SENSOR_COLUMNS

app = FastAPI(title="ml-models")

TWIN_CORE_URL = "http://localhost:8001/telemetry/latest"


def frame_to_feature_dict(frame: dict) -> dict:
    """
    Flatten a telemetry_frame's sensors into the flat dict the models
    expect. NOTE: this is a simplified stand-in for the rolling-window
    features used at training time (features.py::add_rolling_features) —
    swap in a real rolling buffer keyed by unit_id before relying on this
    for anything beyond a smoke test.
    """
    feats = dict(frame["sensors"])
    for col in SENSOR_COLUMNS:
        feats[f"{col}_roll_mean"] = feats[col]
        feats[f"{col}_roll_std"] = 0.0
    return feats


@app.get("/health/latest")
def latest():
    frame = requests.get(TWIN_CORE_URL, timeout=5).json()
    feats = frame_to_feature_dict(frame)

    anomaly = anomaly_detector.score(feats)
    rul = rul_model.predict(feats)
    fault_probs = fault_classifier.predict_proba(feats)

    top_fault = max(fault_probs, key=fault_probs.get)
    advisory = (
        f"Elevated risk of {top_fault} (RUL ~{rul:.0f} cycles) — recommend inspection."
        if anomaly["is_anomalous"] and top_fault != "none"
        else "Nominal."
    )

    return {
        "unit_id": frame["unit_id"],
        "cycle": frame["cycle"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "anomaly": anomaly,
        "rul_estimate_cycles": rul,
        "fault_probabilities": fault_probs,
        "advisory": advisory,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
