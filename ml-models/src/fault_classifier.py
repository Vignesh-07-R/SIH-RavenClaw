"""
Baseline fault-mode classifier: Random Forest over rolling sensor features,
predicting which of FAULT_MODES is developing (or "none"). Includes basic
feature importances as a first step toward explainability.
"""

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from features import add_rolling_features, feature_columns, load_dataset

MODEL_PATH = Path(__file__).parent / "fault_classifier.joblib"


def train() -> RandomForestClassifier:
    df = add_rolling_features(load_dataset())
    cols = feature_columns(df)

    X_train, X_test, y_train, y_test = train_test_split(
        df[cols], df["fault_mode"], test_size=0.2, random_state=42, stratify=df["fault_mode"]
    )

    model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)

    print(classification_report(y_test, model.predict(X_test)))

    joblib.dump({"model": model, "columns": cols}, MODEL_PATH)
    print(f"Saved to {MODEL_PATH}")
    return model


def predict_proba(frame_features: dict) -> dict:
    bundle = joblib.load(MODEL_PATH)
    model, cols = bundle["model"], bundle["columns"]
    x = [[frame_features.get(c, 0.0) for c in cols]]
    probs = model.predict_proba(x)[0]
    return dict(zip(model.classes_, (float(p) for p in probs)))


if __name__ == "__main__":
    train()
