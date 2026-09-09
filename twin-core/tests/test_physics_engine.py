import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))

from mission_profile import MISSION_PROFILE, build_rpm_and_phase_trace  # noqa: E402
from physics_engine import BASELINE, breach_flags, physics_informed_readings  # noqa: E402


def test_physics_informed_readings_at_idle():
    readings = physics_informed_readings(rpm=BASELINE["rpm_idle"], ambient_c=25.0)
    # at idle, rpm_frac clamps to 0, so cht/egt should sit near their floor values
    assert readings["rpm"] == BASELINE["rpm_idle"]
    assert 70 <= readings["cht_c"] < 90
    assert 390 <= readings["egt_c"] < 420


def test_physics_informed_readings_at_max_continuous_rpm():
    readings = physics_informed_readings(rpm=BASELINE["rpm_max_cont"], ambient_c=25.0)
    # at max RPM, rpm_frac clamps to 1, so cht/egt should be near their normal (not fault) ceiling
    assert abs(readings["cht_c"] - (BASELINE["cht_normal_c"] + 25 * 0.15)) < 1e-6
    assert abs(readings["egt_c"] - BASELINE["egt_normal_c"]) < 1e-6


def test_physics_informed_readings_clamps_beyond_max_rpm():
    # RPM past max_cont shouldn't push sensors past the max_cont values --
    # rpm_frac clamps to 1.0 rather than extrapolating.
    at_max = physics_informed_readings(rpm=BASELINE["rpm_max_cont"], ambient_c=25.0)
    beyond_max = physics_informed_readings(rpm=BASELINE["rpm_max_cont"] + 2000, ambient_c=25.0)
    assert at_max["cht_c"] == beyond_max["cht_c"]
    assert at_max["egt_c"] == beyond_max["egt_c"]


def test_mission_profile_trace_shape():
    rpm_trace, phase_trace = build_rpm_and_phase_trace()
    expected_len = sum(duration for _, duration, _ in MISSION_PROFILE)
    assert len(rpm_trace) == expected_len
    assert len(phase_trace) == expected_len
    # first phase should start at idle rpm, ramping toward the first target
    assert rpm_trace[0] == BASELINE["rpm_idle"]
    assert phase_trace[0] == MISSION_PROFILE[0][0]


def test_breach_flags_detects_overheat():
    frame = {
        "sensors": {"cht_c": 160, "egt_c": 850, "oil_press_psi": 55, "oil_temp_c": 100},
        "limits": {"cht_max_c": 150, "egt_max_c": 900, "oil_press_min_psi": 22,
                   "oil_press_max_psi": 72, "oil_temp_max_c": 120},
    }
    flags = breach_flags(frame)
    assert flags["cht_over"] is True
    assert flags["egt_over"] is False


def test_breach_flags_detects_low_oil_pressure():
    frame = {
        "sensors": {"cht_c": 100, "egt_c": 700, "oil_press_psi": 15, "oil_temp_c": 90},
        "limits": {"cht_max_c": 150, "egt_max_c": 900, "oil_press_min_psi": 22,
                   "oil_press_max_psi": 72, "oil_temp_max_c": 120},
    }
    flags = breach_flags(frame)
    assert flags["oil_press_low"] is True
    assert flags["oil_press_high"] is False


def test_breach_flags_uses_module_default_limits_when_omitted():
    healthy_frame = {"sensors": physics_informed_readings(BASELINE["rpm_cruise"], 25.0)}
    flags = breach_flags(healthy_frame)
    assert not any(flags.values())