"""
Physics-informed engine model for twin-core.

This is the "thermodynamic behavior model" the PS asks for (section A/B):
given an RPM reading, derive the dependent sensor values (CHT, EGT, fuel
flow, oil pressure/temp, vibration) instead of sampling them independently
-- a real piston engine's sensors move together because they're all driven
by the same combustion process, not by eight unrelated random processes.

Also owns the operating-limit definitions and the breach-checking logic,
since "is this reading outside a safe limit" is a physics/domain concern,
not an API or streaming concern -- api/twin_api.py and simulator/
live_stream.py both import breach_flags() from here instead of each
re-implementing it.

Baseline parameters are read from twin-core/config/engine_params.yaml
(Rotax 912-class) rather than hardcoded, so tuning the engine model means
editing one YAML file, not hunting through multiple .py files.
"""

from pathlib import Path
from typing import Dict

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "engine_params.yaml"


def _load_baseline() -> dict:
    import yaml
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


BASELINE = _load_baseline()

# Matches the "limits" block shape in docs/DATA_CONTRACT.md section 2 --
# both twin_api.py and live_stream.py attach this straight onto a frame.
LIMITS = {
    "cht_max_c": BASELINE["cht_max_c"],
    "egt_max_c": BASELINE["egt_max_c"],
    "oil_press_min_psi": BASELINE["oil_press_min_psi"],
    "oil_press_max_psi": BASELINE["oil_press_max_psi"],
    "oil_temp_max_c": BASELINE["oil_temp_max_c"],
}

PSI_TO_BAR = 0.0689476


def physics_informed_readings(rpm: float, ambient_c: float = 25.0) -> Dict[str, float]:
    """
    Derive dependent sensor values from a single RPM reading.

    rpm_frac is 0 at idle and 1 at max continuous RPM; every sensor is a
    simple function of that one load fraction, plus ambient temperature
    where it matters (CHT). This is deliberately a simplified relation,
    not a full combustion thermodynamics model -- refine it here if the
    team wants a more faithful physics layer; every caller (simulator,
    tests) picks up the change automatically.
    """
    rpm_frac = (rpm - BASELINE["rpm_idle"]) / (BASELINE["rpm_max_cont"] - BASELINE["rpm_idle"])
    rpm_frac = min(max(rpm_frac, 0.0), 1.0)

    return {
        "rpm": rpm,
        "cht_c": 70 + rpm_frac * (BASELINE["cht_normal_c"] - 70) + ambient_c * 0.15,
        "egt_c": 400 + rpm_frac * (BASELINE["egt_normal_c"] - 400),
        "fuel_flow_lph": 4 + rpm_frac * (BASELINE["fuel_flow_cruise_lph"] - 4),
        "oil_press_psi": BASELINE["oil_press_normal_psi"] - (1 - rpm_frac) * 10,
        "oil_temp_c": 60 + rpm_frac * (BASELINE["oil_temp_normal_c"] - 60),
        "vibration_g": BASELINE["vibration_normal_g"] * (0.5 + 0.5 * rpm_frac),
        "map_inhg": BASELINE["map_idle_inhg"] + rpm_frac * (BASELINE["map_max_inhg"] - BASELINE["map_idle_inhg"]),
        "ambient_c": ambient_c,
        # NOTE: there's no independent throttle input in this simplified
        # model -- the mission profile drives target RPM directly. This
        # approximates throttle position as the same load fraction that
        # drives every other sensor, which is backwards from a real engine
        # (throttle causes RPM, not the reverse) but is a reasonable stand-in
        # until the simulator models throttle as its own input. Revisit if
        # Group 3's animation needs throttle to lead RPM rather than track it.
        "throttle_pct": round(rpm_frac * 100, 1),
    }


def lag_step(previous: float, raw_target: float, alpha: float) -> float:
    """
    One step of an exponential (first-order) lag filter:
        new = alpha * raw_target + (1 - alpha) * previous

    Models thermal inertia -- CHT and oil temperature don't jump
    instantly to a new equilibrium the way EGT/oil-pressure/vibration
    effectively do; they climb toward it over many cycles. Lower alpha =
    slower/heavier response.

    This is intentionally a single stateless step, not a batch/array
    operation: the caller (EngineUnitSimulator) owns the running lagged
    value as persistent per-unit state and calls this once per cycle,
    *after* any fault has been added to raw_target -- a developing fault's
    heat buildup has the same thermal inertia as normal operation, so it
    should never bypass the lag by being added after it.
    """
    return alpha * raw_target + (1 - alpha) * previous


def to_engine_state(sensors: dict) -> Dict[str, float]:
    """
    Convert a telemetry_frame's `sensors` dict into the reduced,
    animation-friendly `engine_state` block the 3D viewer (Group 3) consumes.
    Different unit choices (bar instead of psi) and field names match what
    the animation code expects, not the internal sensor schema -- keep this
    function as the one place that translation happens.
    """
    return {
        "rpm": round(sensors["rpm"], 1),
        "throttle": round(sensors.get("throttle_pct", 0.0), 1),
        "egt_c": round(sensors["egt_c"], 1),
        "cht_c": round(sensors["cht_c"], 1),
        "oil_pressure_bar": round(sensors["oil_press_psi"] * PSI_TO_BAR, 2),
        "oil_temperature_c": round(sensors["oil_temp_c"], 1),
        "fuel_flow_lph": round(sensors["fuel_flow_lph"], 2),
        "vibration_g": round(sensors["vibration_g"], 3),
    }


def breach_flags(frame: dict) -> Dict[str, bool]:
    """
    Flag sensor readings past the Rotax 912-class limits. `frame` needs a
    "sensors" dict; an optional "limits" dict overrides the defaults here
    (useful for testing against a different engine's limits without
    touching the module-level constant).
    """
    sensors = frame["sensors"]
    limits = frame.get("limits", LIMITS)
    return {
        "cht_over": sensors["cht_c"] > limits["cht_max_c"],
        "egt_over": sensors["egt_c"] > limits["egt_max_c"],
        "oil_press_low": sensors["oil_press_psi"] < limits["oil_press_min_psi"],
        "oil_press_high": sensors["oil_press_psi"] > limits["oil_press_max_psi"],
        "oil_temp_over": sensors["oil_temp_c"] > limits["oil_temp_max_c"],
    }