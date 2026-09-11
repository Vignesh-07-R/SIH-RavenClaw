"""
Fault injection for twin-core.

data/engine_data_generator.py bakes a fault into a whole unit's CSV ahead of
time (fixed onset cycle, fixed severity curve) — great for training data,
useless for a live demo where you want to walk up to the dashboard and
*trigger* a fault to show the system catching it in real time.

This module ports the same fault relations but applies them frame-by-frame
against a `progress` value (0 = healthy, 1 = failed) that something else
controls — e.g. live_stream.py incrementing it each cycle after an operator
hits "inject fault" via the API.
"""

import random

FAULT_MODES = ["none", "overheating", "misfire_vibration", "oil_degradation", "sensor_drift"]


def apply_fault(fault_mode: str, progress: float, sensors: dict) -> dict:
    """
    Apply one fault mode's degradation to a single telemetry_frame's sensors,
    given how far into the fault we are (0.0 healthy -> 1.0 failure point).

    Returns a *new* sensors dict — does not mutate the input, so callers can
    keep the clean baseline around for comparison (e.g. logging what the
    fault injector changed, for a "ground truth" panel in the demo).
    """
    if fault_mode not in FAULT_MODES:
        raise ValueError(f"unknown fault_mode: {fault_mode!r}, expected one of {FAULT_MODES}")

    s = dict(sensors)  # shallow copy — don't mutate caller's frame
    progress = min(max(progress, 0.0), 1.0)

    if fault_mode == "overheating":
        s["cht_c"] = s["cht_c"] + progress * 45
        s["egt_c"] = s["egt_c"] + progress * 60

    elif fault_mode == "misfire_vibration":
        noise = random.gauss(0, 0.4) if progress > 0.7 else 0.0
        s["vibration_g"] = s["vibration_g"] + progress * 3.5 + noise

    elif fault_mode == "oil_degradation":
        s["oil_press_psi"] = s["oil_press_psi"] - progress * 30
        s["oil_temp_c"] = s["oil_temp_c"] + progress * 25

    elif fault_mode == "sensor_drift":
        # one sensor silently biases high -- which sensor is resolved once
        # per injected fault (not per frame), so the drift stays on the same
        # channel for the whole mission until reset_drift_state() is called.
        # Picking from three plausible sensors (rather than only ever EGT)
        # matches data/engine_data_generator.py's sensor_drift fault.
        if not hasattr(apply_fault, "_drift_sensor"):
            apply_fault._drift_sensor = random.choice(["egt_c", "cht_c", "oil_press_psi"])
        drift_sensor = apply_fault._drift_sensor
        if drift_sensor == "egt_c":
            s["egt_c"] = s["egt_c"] + progress * 80
        elif drift_sensor == "cht_c":
            s["cht_c"] = s["cht_c"] + progress * 30
        elif drift_sensor == "oil_press_psi":
            s["oil_press_psi"] = s["oil_press_psi"] + progress * 20

    # fault_mode == "none": no change

    return s


def reset_drift_state() -> None:
    """Call this when starting a fresh mission/demo so sensor_drift's
    sensor choice re-randomizes instead of reusing the last run's pick."""
    if hasattr(apply_fault, "_drift_sensor"):
        del apply_fault._drift_sensor


if __name__ == "__main__":
    healthy = {
        "rpm": 5000.0, "cht_c": 110.0, "egt_c": 800.0, "fuel_flow_lph": 15.0,
        "oil_press_psi": 58.0, "oil_temp_c": 100.0, "vibration_g": 0.5, "ambient_c": 25.0,
    }
    for progress in (0.0, 0.5, 1.0):
        print(f"overheating @ progress={progress}:", apply_fault("overheating", progress, healthy))