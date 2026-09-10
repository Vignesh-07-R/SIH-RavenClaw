"""
rul_model.py — RUL (Remaining Useful Life) regression (ml-models, Group 2).

Trains a Random Forest regressor to predict RUL (cycles remaining) from
the rolling features across all sensors. RUL is the ground-truth label
column in the dataset — used as the training TARGET, never as an input
feature.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

from features import add_rolling_features, feature_columns, load_dataset

ARTIFACT_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "rul_model.joblib"


def train(df: pd.DataFrame = None) -> RandomForestRegressor:
    if df is None:
        df = add_rolling_features(load_dataset())

    cols = feature_columns(df)

    # split by unit_id (not by row) so the same engine run never leaks
    # across train/test
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(df, groups=df["unit_id"]))
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

    model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(train_df[cols], train_df["RUL"])

    preds = model.predict(test_df[cols])
    mae = mean_absolute_error(test_df["RUL"], preds)
    print(f"Held-out MAE: {mae:.2f} cycles (RUL is capped at 200)")

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_cols": cols}, ARTIFACT_PATH)
    return model


def load_model():
    bundle = joblib.load(ARTIFACT_PATH)
    return bundle["model"], bundle["feature_cols"]


def predict(df: pd.DataFrame) -> pd.DataFrame:
    """Adds rul_estimate_cycles (int) to the DataFrame."""
    model, cols = load_model()
    df = df.copy()
    df["rul_estimate_cycles"] = np.round(model.predict(df[cols])).astype(int)
    return df


if __name__ == "__main__":
    raw = load_dataset()
    featured = add_rolling_features(raw)

    print("Training RUL regressor...")
    train(featured)
    print(f"Saved model to {ARTIFACT_PATH}")

    scored = predict(featured)
    print("\nSample predictions vs ground truth (last 5 rows of unit 1):")
    unit1 = scored[scored["unit_id"] == 1].tail(5)
    print(unit1[["cycle", "RUL", "rul_estimate_cycles"]].to_string(index=False))
