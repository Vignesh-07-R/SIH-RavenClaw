"""
Baseline unsupervised anomaly detector: Isolation Forest over rolling
sensor features. Trained only on `fault_mode == "none"` cycles, so
"anomalous" means "doesn't look like healthy operation" — no fault labels
needed at train time, which mirrors a real deployment.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

from features import add_rolling_features, feature_columns, load_dataset

MODEL_PATH = Path(__file__).parent / "anomaly_model.joblib"


def train() -> IsolationForest:
    df = add_rolling_features(load_dataset())
    healthy = df[df["fault_mode"] == "none"]
    cols = feature_columns(df)

    model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    model.fit(healthy[cols])
    joblib.dump({"model": model, "columns": cols}, MODEL_PATH)
    print(f"Trained on {len(healthy)} healthy rows, saved to {MODEL_PATH}")
    return model


def score(frame_features: dict) -> dict:
    """frame_features: dict of {column_name: value} matching training columns."""
    bundle = joblib.load(MODEL_PATH)
    model, cols = bundle["model"], bundle["columns"]
    x = pd.DataFrame([[frame_features.get(c, 0.0) for c in cols]], columns=cols)
    raw_score = -model.score_samples(x)[0]  # higher = more anomalous
    is_anomalous = model.predict(x)[0] == -1
    return {"is_anomalous": bool(is_anomalous), "score": float(raw_score)}


if __name__ == "__main__":
    train()
