"""
features.py — shared feature engineering for ml-models (Group 2).

Interface contract (see tests/test_features.py):
    load_dataset(path=None)      -> raw telemetry DataFrame
    add_rolling_features(df)     -> same df + <sensor>_roll_mean / _roll_std
    feature_columns(df)          -> list of engineered feature column names

Rolling features are computed PER unit_id (never across different engine
runs) so the start of one unit's mission never blends with the tail of
another. min_periods=1 and ddof=0 guarantee no NaNs even on the first
row of each run.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Sensor columns per docs/DATA_CONTRACT.md section 1.
# fault_mode and RUL are labels, never fed in as features.
SENSOR_COLUMNS = [
    "rpm", "cht_c", "egt_c", "fuel_flow_lph",
    "oil_press_psi", "oil_temp_c", "vibration_g", "ambient_c",
    "battery_voltage_v", "alternator_current_a",
    "injection_timing_btdc_deg", "injector_pulse_width_ms",
]

LABEL_COLUMNS = ["fault_mode", "RUL"]
ID_COLUMNS = ["unit_id", "cycle", "phase"]

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "engine_telemetry_dataset.csv"

ROLL_WINDOW = 30  # cycles


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw per-cycle telemetry CSV (data/engine_telemetry_dataset.csv)."""
    path = Path(path) if path else DEFAULT_DATA_PATH
    df = pd.read_csv(path)
    return df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)


def add_rolling_features(df: pd.DataFrame, window: int = ROLL_WINDOW) -> pd.DataFrame:
    """
    Add <sensor>_roll_mean / <sensor>_roll_std columns, computed per
    unit_id with min_periods=1 so every row (including the first of each
    run) gets a value — no NaNs.
    """
    df = df.copy()
    grouped = df.groupby("unit_id", sort=False)

    for col in SENSOR_COLUMNS:
        if col not in df.columns:
            continue
        roll = grouped[col].rolling(window=window, min_periods=1)
        df[f"{col}_roll_mean"] = roll.mean().reset_index(level=0, drop=True)
        # ddof=0 (population std) avoids NaN on single-row windows,
        # unlike pandas' default ddof=1 sample std.
        df[f"{col}_roll_std"] = roll.std(ddof=0).reset_index(level=0, drop=True).fillna(0.0)

    return df


def feature_columns(df: pd.DataFrame) -> list:
    """Return only the engineered rolling-feature columns (model inputs).
    Excludes id columns, raw sensor columns, and label columns."""
    return [c for c in df.columns if c.endswith("_roll_mean") or c.endswith("_roll_std")]


def stratified_group_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split by unit_id (so no engine run leaks across train/test) while
    guaranteeing every fault_mode gets proportional representation in
    both splits. Plain GroupShuffleSplit does NOT guarantee this — with
    few units per fault type (e.g. 8), it can randomly place an entire
    fault type's units into train only, leaving test with 0 samples for
    that class.

    Each unit is assigned a "primary_fault" = the most common non-"none"
    fault_mode value it contains (or "none" if it never faults), then
    units are split proportionally within each primary_fault group.
    """
    rng = np.random.default_rng(random_state)

    def primary_fault(unit_df):
        non_none = unit_df.loc[unit_df["fault_mode"] != "none", "fault_mode"]
        return non_none.mode().iloc[0] if len(non_none) else "none"

    unit_labels = df.groupby("unit_id").apply(primary_fault, include_groups=False)

    train_units, test_units = [], []
    for label, group in unit_labels.groupby(unit_labels):
        units = group.index.to_numpy().copy()
        rng.shuffle(units)
        n_test = max(1, round(len(units) * test_size)) if len(units) > 1 else 0
        test_units.extend(units[:n_test])
        train_units.extend(units[n_test:])

    train_df = df[df["unit_id"].isin(train_units)]
    test_df = df[df["unit_id"].isin(test_units)]
    return train_df, test_df


if __name__ == "__main__":
    raw = load_dataset()
    print(f"Loaded {len(raw)} rows across {raw['unit_id'].nunique()} units")

    featured = add_rolling_features(raw)
    cols = feature_columns(featured)
    print(f"Added {len(cols)} rolling feature columns:")
    print(cols)

    assert featured[cols].isna().sum().sum() == 0, "Found NaNs in engineered features!"
    print("\nNo NaNs in engineered features — OK")

    out_path = DEFAULT_DATA_PATH.parent / "engine_features_rolling.csv"
    featured.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")