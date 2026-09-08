import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features import add_rolling_features, feature_columns, load_dataset  # noqa: E402


def test_rolling_features_present():
    df = add_rolling_features(load_dataset().head(200))
    cols = feature_columns(df)
    assert "cht_c_roll_mean" in cols
    assert "cht_c_roll_std" in cols
    assert not df["cht_c_roll_mean"].isna().any()
