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
        "ambient_c": ambient_c,
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