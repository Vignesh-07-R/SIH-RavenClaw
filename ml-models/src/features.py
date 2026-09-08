"""
Shared feature extraction so anomaly_detector, rul_model, and
fault_classifier all train/predict on the same representation.
"""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "engine_telemetry_dataset.csv"

SENSOR_COLUMNS = [
    "rpm", "cht_c", "egt_c", "fuel_flow_lph",
    "oil_press_psi", "oil_temp_c", "vibration_g", "ambient_c",
]


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def add_rolling_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Add rolling mean/std per sensor, per unit — degradation trends (e.g.
    slowly rising CHT) show up more clearly in a rolling window than in a
    single instantaneous reading.
    """
    df = df.sort_values(["unit_id", "cycle"]).copy()
    grouped = df.groupby("unit_id")[SENSOR_COLUMNS]
    for col in SENSOR_COLUMNS:
        df[f"{col}_roll_mean"] = grouped[col].transform(lambda s: s.rolling(window, min_periods=1).mean())
        df[f"{col}_roll_std"] = grouped[col].transform(lambda s: s.rolling(window, min_periods=1).std().fillna(0))
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith("_roll_mean") or c.endswith("_roll_std")] + SENSOR_COLUMNS


if __name__ == "__main__":
    df = add_rolling_features(load_dataset())
    print(df[feature_columns(df)].head())
