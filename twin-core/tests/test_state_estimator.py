import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from state_estimator import to_telemetry_frame, breach_flags  # noqa: E402


def test_to_telemetry_frame_shape():
    row = {
        "unit_id": 1, "cycle": 0, "phase": "idle", "rpm": 1400.0,
        "cht_c": 90.0, "egt_c": 500.0, "fuel_flow_lph": 4.0,
        "oil_press_psi": 55.0, "oil_temp_c": 80.0, "vibration_g": 0.3,
        "ambient_c": 25.0,
    }
    frame = to_telemetry_frame(row)
    assert frame["unit_id"] == 1
    assert set(frame["sensors"]) == {
        "rpm", "cht_c", "egt_c", "fuel_flow_lph",
        "oil_press_psi", "oil_temp_c", "vibration_g", "ambient_c",
    }


def test_breach_flags_detects_overheat():
    frame = {
        "sensors": {"cht_c": 160, "egt_c": 850, "oil_press_psi": 55, "oil_temp_c": 100},
        "limits": {"cht_max_c": 150, "egt_max_c": 900, "oil_press_min_psi": 22,
                   "oil_press_max_psi": 72, "oil_temp_max_c": 120},
    }
    flags = breach_flags(frame)
    assert flags["cht_over"] is True
    assert flags["egt_over"] is False
