"""
Synthetic Aero Piston Engine Digital Twin Data Generator
==========================================================
Built for SIH26054 (AI-Enabled Digital Twin for MALE UAV Aero Piston Engines)

Grounded in two real references (no real UAV engine telemetry is public,
so this generator simulates plausible data instead of inventing numbers):

1. Baseline operating parameters -> Rotax 912-class engine specs
   (the engine family actually used in several MALE-class UAVs).
2. Dataset shape / RUL labeling convention -> NASA C-MAPSS turbofan
   degradation benchmark (engine_id, cycle, settings, sensors, RUL).
   We reuse the *structure*, not the data, since C-MAPSS is a jet
   engine, not a piston engine.

Output: a CSV per "engine unit", each a full mission from start-up to
either a safe landing (healthy unit) or a developing fault (faulty unit),
in the same run-to-failure style C-MAPSS uses for RUL model training.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)  # fixed seed -> reproducible dataset

# ---------------------------------------------------------------------------
# 1. Baseline engine parameters (Rotax 912-class, from real operator specs)
# ---------------------------------------------------------------------------
BASELINE = {
    "rpm_idle": 1400,
    "rpm_cruise": 5000,
    "rpm_max_cont": 5500,
    "cht_normal_c": 110,
    "cht_max_c": 150,
    "egt_normal_c": 800,
    "egt_max_c": 900,
    "oil_press_normal_psi": 58,
    "oil_press_min_psi": 22,
    "oil_press_max_psi": 72,
    "oil_temp_normal_c": 100,
    "oil_temp_max_c": 120,
    "fuel_flow_cruise_lph": 15.0,
    "vibration_normal_g": 0.5,
}

# Mission phases as (phase_name, duration_cycles, rpm_target) tuples.
# One "cycle" = one telemetry sample (e.g. 1 Hz), matching C-MAPSS's
# per-cycle sensor snapshot convention.
MISSION_PROFILE = [
    ("idle", 30, BASELINE["rpm_idle"]),
    ("climb", 60, BASELINE["rpm_max_cont"]),
    ("cruise", 300, BASELINE["rpm_cruise"]),
    ("descent", 60, BASELINE["rpm_idle"] + 800),
    ("idle", 20, BASELINE["rpm_idle"]),
]

FAULT_MODES = ["none", "overheating", "misfire_vibration", "oil_degradation", "sensor_drift"]


def _physics_informed_readings(rpm: np.ndarray, ambient_c: float = 25.0):
    """
    Derive dependent sensor values from RPM using simplified physics
    relations, instead of sampling every sensor independently. This is
    the "physics-informed" layer the PS explicitly asks for: EGT/CHT/fuel
    flow are not random -- they scale with engine load (RPM), the way a
    real piston engine behaves.
    """
    rpm_frac = (rpm - BASELINE["rpm_idle"]) / (BASELINE["rpm_max_cont"] - BASELINE["rpm_idle"])
    rpm_frac = np.clip(rpm_frac, 0, 1)

    cht = 70 + rpm_frac * (BASELINE["cht_normal_c"] - 70) + ambient_c * 0.15
    egt = 400 + rpm_frac * (BASELINE["egt_normal_c"] - 400)
    fuel_flow = 4 + rpm_frac * (BASELINE["fuel_flow_cruise_lph"] - 4)
    oil_press = BASELINE["oil_press_normal_psi"] - (1 - rpm_frac) * 10  # lower at idle
    oil_temp = 60 + rpm_frac * (BASELINE["oil_temp_normal_c"] - 60)
    vibration = BASELINE["vibration_normal_g"] * (0.5 + 0.5 * rpm_frac)

    return cht, egt, fuel_flow, oil_press, oil_temp, vibration


def _apply_fault(fault_mode: str, progress: np.ndarray, cht, egt, oil_press, oil_temp, vibration):
    """
    Inject a gradual degradation trend for one fault mode.
    `progress` is 0 -> 1 across the unit's life (0 = healthy start,
    1 = failure point), mirroring C-MAPSS's run-to-failure trajectories.
    """
    if fault_mode == "overheating":
        cht = cht + progress * 45          # trends toward/past 150C limit
        egt = egt + progress * 60
    elif fault_mode == "misfire_vibration":
        vibration = vibration + progress * 3.5 + (progress > 0.7) * RNG.normal(0, 0.4, len(progress))
    elif fault_mode == "oil_degradation":
        oil_press = oil_press - progress * 30   # drifts toward the 22 psi minimum
        oil_temp = oil_temp + progress * 25
    elif fault_mode == "sensor_drift":
        egt = egt + progress * 80 * (RNG.random() > 0.5)  # one sensor biases high
    return cht, egt, oil_press, oil_temp, vibration


def generate_unit(unit_id: int, fault_mode: str = "none", ambient_c: float = 25.0) -> pd.DataFrame:
    """Generate one full-mission time series for a single engine unit."""
    rpm_trace, phase_trace = [], []
    for phase, duration, rpm_target in MISSION_PROFILE:
        # smooth ramp toward each phase's target RPM instead of a step change
        start_rpm = rpm_trace[-1] if rpm_trace else BASELINE["rpm_idle"]
        ramp = np.linspace(start_rpm, rpm_target, duration)
        rpm_trace.extend(ramp)
        phase_trace.extend([phase] * duration)

    rpm = np.array(rpm_trace) + RNG.normal(0, 15, len(rpm_trace))  # sensor noise
    n_cycles = len(rpm)

    cht, egt, fuel_flow, oil_press, oil_temp, vibration = _physics_informed_readings(rpm, ambient_c)

    # progress toward failure: 0 for a healthy unit, 0->1 for a faulty one
    if fault_mode == "none":
        progress = np.zeros(n_cycles)
    else:
        onset = int(n_cycles * RNG.uniform(0.3, 0.5))  # fault starts partway through
        progress = np.clip((np.arange(n_cycles) - onset) / (n_cycles - onset), 0, 1)

    cht, egt, oil_press, oil_temp, vibration = _apply_fault(
        fault_mode, progress, cht, egt, oil_press, oil_temp, vibration
    )

    # sensor noise on top of the physics + fault signal, like C-MAPSS's injected noise
    cht += RNG.normal(0, 1.5, n_cycles)
    egt += RNG.normal(0, 5, n_cycles)
    oil_press += RNG.normal(0, 1.0, n_cycles)
    oil_temp += RNG.normal(0, 1.0, n_cycles)
    vibration += RNG.normal(0, 0.05, n_cycles)

    rul = (n_cycles - np.arange(n_cycles)) if fault_mode != "none" else np.full(n_cycles, 999)
    rul = np.minimum(rul, 200)  # cap RUL, same convention C-MAPSS uses for stable training

    return pd.DataFrame({
        "unit_id": unit_id,
        "cycle": np.arange(n_cycles),
        "phase": phase_trace,
        "rpm": rpm,
        "cht_c": cht,
        "egt_c": egt,
        "fuel_flow_lph": fuel_flow,
        "oil_press_psi": oil_press,
        "oil_temp_c": oil_temp,
        "vibration_g": vibration,
        "ambient_c": ambient_c,
        "fault_mode": fault_mode,
        "RUL": rul,
    })


def generate_dataset(n_healthy: int = 20, n_faulty_per_mode: int = 8) -> pd.DataFrame:
    """Generate a full multi-unit dataset, mixing healthy and faulty units."""
    units = []
    unit_id = 1
    for _ in range(n_healthy):
        ambient = RNG.uniform(15, 40)  # varying environmental conditions
        units.append(generate_unit(unit_id, "none", ambient))
        unit_id += 1
    for fault_mode in FAULT_MODES[1:]:
        for _ in range(n_faulty_per_mode):
            ambient = RNG.uniform(15, 40)
            units.append(generate_unit(unit_id, fault_mode, ambient))
            unit_id += 1
    return pd.concat(units, ignore_index=True)


if __name__ == "__main__":
    import os

    out_path = os.path.join(os.path.dirname(__file__), "engine_telemetry_dataset.csv")
    df = generate_dataset()
    df.to_csv(out_path, index=False)
    print(f"Generated {df['unit_id'].nunique()} units, {len(df)} total rows -> {out_path}")
    print(df.groupby("fault_mode")["unit_id"].nunique())
