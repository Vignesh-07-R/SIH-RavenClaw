import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))

from fault_injector import FAULT_MODES, apply_fault, reset_drift_state  # noqa: E402

HEALTHY_SENSORS = {
    "rpm": 5000.0, "cht_c": 110.0, "egt_c": 800.0, "fuel_flow_lph": 15.0,
    "oil_press_psi": 58.0, "oil_temp_c": 100.0, "vibration_g": 0.5, "ambient_c": 25.0,
}


def test_apply_fault_rejects_unknown_mode():
    try:
        apply_fault("not_a_real_fault", 0.5, HEALTHY_SENSORS)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_apply_fault_none_leaves_sensors_unchanged():
    result = apply_fault("none", 1.0, HEALTHY_SENSORS)
    assert result == HEALTHY_SENSORS


def test_apply_fault_does_not_mutate_input():
    original = dict(HEALTHY_SENSORS)
    apply_fault("overheating", 0.8, HEALTHY_SENSORS)
    assert HEALTHY_SENSORS == original


def test_overheating_increases_with_progress():
    at_zero = apply_fault("overheating", 0.0, HEALTHY_SENSORS)
    at_half = apply_fault("overheating", 0.5, HEALTHY_SENSORS)
    at_full = apply_fault("overheating", 1.0, HEALTHY_SENSORS)
    assert at_zero["cht_c"] < at_half["cht_c"] < at_full["cht_c"]
    assert at_zero["egt_c"] < at_half["egt_c"] < at_full["egt_c"]


def test_oil_degradation_drops_pressure_and_raises_temp():
    result = apply_fault("oil_degradation", 1.0, HEALTHY_SENSORS)
    assert result["oil_press_psi"] < HEALTHY_SENSORS["oil_press_psi"]
    assert result["oil_temp_c"] > HEALTHY_SENSORS["oil_temp_c"]


def test_misfire_vibration_increases_with_progress():
    at_zero = apply_fault("misfire_vibration", 0.0, HEALTHY_SENSORS)
    at_full = apply_fault("misfire_vibration", 1.0, HEALTHY_SENSORS)
    assert at_zero["vibration_g"] < at_full["vibration_g"]


def test_sensor_drift_direction_stable_across_calls_until_reset():
    reset_drift_state()
    first = apply_fault("sensor_drift", 0.5, HEALTHY_SENSORS)
    second = apply_fault("sensor_drift", 0.5, HEALTHY_SENSORS)
    # same progress, same coin-flip outcome -> identical result until reset
    assert first["egt_c"] == second["egt_c"]
    reset_drift_state()


def test_progress_is_clamped_to_zero_one_range():
    over_one = apply_fault("overheating", 1.5, HEALTHY_SENSORS)
    at_one = apply_fault("overheating", 1.0, HEALTHY_SENSORS)
    assert over_one["cht_c"] == at_one["cht_c"]


def test_all_declared_fault_modes_are_handled():
    # every mode in FAULT_MODES should run without raising
    for mode in FAULT_MODES:
        reset_drift_state()
        apply_fault(mode, 0.5, HEALTHY_SENSORS)