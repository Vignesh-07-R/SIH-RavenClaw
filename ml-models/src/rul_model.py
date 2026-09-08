"""
Baseline RUL regressor: Gradient Boosting over rolling sensor features,
trained across all units (healthy units are correctly labeled RUL=999,
capped consistently with the faulty units' cap in the generator).
"""

from pathlib import Path

import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from features import add_rolling_features, feature_columns, load_dataset

MODEL_PATH = Path(__file__).parent / "rul_model.joblib"


def train() -> GradientBoostingRegressor:
    df = add_rolling_features(load_dataset())
    cols = feature_columns(df)

    X_train, X_test, y_train, y_test = train_test_split(
        df[cols], df["RUL"], test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f"Held-out MAE: {mae:.1f} cycles")

    joblib.dump({"model": model, "columns": cols}, MODEL_PATH)
    print(f"Saved to {MODEL_PATH}")
    return model


def predict(frame_features: dict) -> float:
    bundle = joblib.load(MODEL_PATH)
    model, cols = bundle["model"], bundle["columns"]
    x = [[frame_features.get(c, 0.0) for c in cols]]
    return float(model.predict(x)[0])


if __name__ == "__main__":
    train()
