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
 
from features import add_rolling_features, feature_columns, load_dataset, stratified_group_split
 
ARTIFACT_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "rul_model.joblib"
 
# The generator uses RUL=999 as a sentinel for "fully healthy, no known
# failure horizon" rather than a real countdown (faulty units count down
# 1-200). Training on the raw mix makes the regressor try to bridge a
# 1-200 range and a 999 spike with nothing in between. Capping at 200
# treats "999" the same as "fully healthy, RUL=200" -- consistent with
# the original data contract and with how healthy units were labeled
# before this generator update.
RUL_CAP = 200
 
 
def train(df: pd.DataFrame = None) -> RandomForestRegressor:
    if df is None:
        df = add_rolling_features(load_dataset())
 
    cols = feature_columns(df)
    df = df.copy()
    df["RUL"] = df["RUL"].clip(upper=RUL_CAP)
 
    # split by unit_id, stratified by each unit's fault type, so every
    # fault mode is represented in both train and test even with only
    # a handful of units per class
    train_df, test_df = stratified_group_split(df, test_size=0.2, random_state=42)
 
    model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(train_df[cols], train_df["RUL"])
 
    preds = model.predict(test_df[cols])
    mae = mean_absolute_error(test_df["RUL"], preds)
    print(f"Held-out MAE: {mae:.2f} cycles (RUL capped at {RUL_CAP})")
 
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
    scored["RUL"] = scored["RUL"].clip(upper=RUL_CAP)
    print("\nSample predictions vs ground truth (last 5 rows of unit 1):")
    unit1 = scored[scored["unit_id"] == 1].tail(5)
    print(unit1[["cycle", "RUL", "rul_estimate_cycles"]].to_string(index=False))
 