"""
Synthetic Aero Piston Engine Digital Twin Data Generator
==========================================================
Built for SIH26054 (AI-Enabled Digital Twin for MALE UAV Aero Piston Engines)

See twin-core/models/physics_engine.py and twin-core/simulator/unit_generator.py
for the canonical live version of this same physics model -- keep fault
relations and thermal-lag behavior in sync between the two if you change one.
"""

import time

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)  # fixed seed -> reproducible dataset
# Separate RNG for the live-streaming demo helper, so calling it doesn't
# perturb generate_dataset()'s reproducibility if both run in one process.
_STREAM_RNG = np.random.default_rng(7)

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
    "map_idle_inhg": 12.0,  # Manifold Absolute Pressure
    "map_max_inhg": 28.0,
    "cht_thermal_alpha": 0.03,
    "oil_temp_thermal_alpha": 0.02,
}

MISSION_PROFILE = [
    ("idle", 30, BASELINE["rpm_idle"]),
    ("climb", 60, BASELINE["rpm_max_cont"]),
    ("cruise", 300, BASELINE["rpm_cruise"]),
    ("descent", 60, BASELINE["rpm_idle"] + 800),
    ("idle", 20, BASELINE["rpm_idle"]),
]

FAULT_MODES = ["none", "overheating", "misfire_vibration", "oil_degradation", "sensor_drift"]


def _apply_thermal_lag(raw_temp: np.ndarray, ambient_c: float, alpha: float) -> np.ndarray:
    """
    Simulate thermal inertia: lower alpha = slower temperature response.

    Cold-starts at ambient_c (an engine begins at ambient temperature and
    warms up over the mission) rather than snapping to the first cycle's
    raw target -- a previous version of this function set lagged[0] =
    raw_temp[0], which skipped the warm-up curve entirely.
    """
    lagged = np.empty_like(raw_temp)
    lagged[0] = ambient_c
    for t in range(1, len(raw_temp)):
        lagged[t] = alpha * raw_temp[t] + (1 - alpha) * lagged[t - 1]
    return lagged


def _physics_informed_readings(rpm: np.ndarray, ambient_c: float = 25.0):
    """
    Derive dependent sensor values from RPM using simplified physics.

    Returns RAW (non-lagged) cht/oil_temp targets -- thermal lag is applied
    later in generate_unit(), *after* _apply_fault(), so a developing
    fault's heat buildup also has thermal inertia instead of jumping
    straight onto an already-smoothed baseline.
    """
    rpm_frac = (rpm - BASELINE["rpm_idle"]) / (BASELINE["rpm_max_cont"] - BASELINE["rpm_idle"])
    rpm_frac = np.clip(rpm_frac, 0, 1)

    raw_cht = 70 + rpm_frac * (BASELINE["cht_normal_c"] - 70) + ambient_c * 0.15
    raw_oil_temp = 60 + rpm_frac * (BASELINE["oil_temp_normal_c"] - 60)
    egt = 400 + rpm_frac * (BASELINE["egt_normal_c"] - 400)
    fuel_flow = 4 + rpm_frac * (BASELINE["fuel_flow_cruise_lph"] - 4)
    oil_press = BASELINE["oil_press_normal_psi"] - (1 - rpm_frac) * 10
    vibration = BASELINE["vibration_normal_g"] * (0.5 + 0.5 * rpm_frac)
    map_inhg = BASELINE["map_idle_inhg"] + rpm_frac * (BASELINE["map_max_inhg"] - BASELINE["map_idle_inhg"])

    return raw_cht, egt, fuel_flow, oil_press, raw_oil_temp, vibration, map_inhg


def _apply_fault(fault_mode: str, progress: np.ndarray, cht, egt, oil_press, oil_temp, vibration,
                  map_inhg, rng: np.random.Generator):
    """Inject a gradual degradation trend for one fault mode, on the RAW
    (pre-lag) temperature targets."""
    if fault_mode == "overheating":
        cht = cht + progress * 45
        egt = egt + progress * 60
    elif fault_mode == "misfire_vibration":
        vibration = vibration + progress * 3.5 + (progress > 0.7) * rng.normal(0, 0.4, len(progress))
    elif fault_mode == "oil_degradation":
        oil_press = oil_press - progress * 30
        oil_temp = oil_temp + progress * 25
    elif fault_mode == "sensor_drift":
        # one sensor drifts high -- resolved once per unit (not per cycle),
        # so the drift stays on the same channel for the whole mission
        drift_sensor = rng.choice(["egt", "cht", "oil_press"])
        if drift_sensor == "egt":
            egt = egt + progress * 80
        elif drift_sensor == "cht":
            cht = cht + progress * 30
        elif drift_sensor == "oil_press":
            oil_press = oil_press + progress * 20

    return cht, egt, oil_press, oil_temp, vibration, map_inhg


def generate_unit(unit_id: int, fault_mode: str = "none", ambient_c: float = 25.0,
                   rng: np.random.Generator = None) -> pd.DataFrame:
    """
    Generate one full-mission time series for a single engine unit.

    `rng` defaults to the module-level RNG (reproducible dataset
    generation); pass a different generator (e.g. from
    stream_live_telemetry) to draw randomness independently, so calling
    both code paths in one process doesn't perturb each other's sequence.
    """
    rng = rng if rng is not None else RNG
    rpm_trace, phase_trace = [], []
    for phase, duration, rpm_target in MISSION_PROFILE:
        start_rpm = rpm_trace[-1] if rpm_trace else BASELINE["rpm_idle"]
        ramp = np.linspace(start_rpm, rpm_target, duration)
        rpm_trace.extend(ramp)
        phase_trace.extend([phase] * duration)

    rpm = np.array(rpm_trace) + rng.normal(0, 15, len(rpm_trace))
    n_cycles = len(rpm)

    cht, egt, fuel_flow, oil_press, oil_temp, vibration, map_inhg = _physics_informed_readings(rpm, ambient_c)

    if fault_mode == "none":
        progress = np.zeros(n_cycles)
    else:
        onset = int(n_cycles * rng.uniform(0.3, 0.5))
        progress = np.clip((np.arange(n_cycles) - onset) / (n_cycles - onset), 0, 1)

    # fault applied to RAW targets first...
    cht, egt, oil_press, oil_temp, vibration, map_inhg = _apply_fault(
        fault_mode, progress, cht, egt, oil_press, oil_temp, vibration, map_inhg, rng
    )

    # ...THEN thermal lag, so the fault's temperature contribution also
    # climbs gradually instead of jumping instantly onto a pre-lagged baseline
    cht = _apply_thermal_lag(cht, ambient_c, BASELINE["cht_thermal_alpha"])
    oil_temp = _apply_thermal_lag(oil_temp, ambient_c, BASELINE["oil_temp_thermal_alpha"])

    # sensor (measurement) noise, added last -- represents instrument
    # jitter, not physical state, so it belongs after lag, not before
    cht += rng.normal(0, 1.5, n_cycles)
    egt += rng.normal(0, 5, n_cycles)
    oil_press += rng.normal(0, 1.0, n_cycles)
    oil_temp += rng.normal(0, 1.0, n_cycles)
    vibration += rng.normal(0, 0.05, n_cycles)
    map_inhg += rng.normal(0, 0.2, n_cycles)

    rul = (n_cycles - np.arange(n_cycles)) if fault_mode != "none" else np.full(n_cycles, 999)
    rul = np.minimum(rul, 200)

    return pd.DataFrame({
        "unit_id": unit_id,
        "cycle": np.arange(n_cycles),
        "phase": phase_trace,
        "rpm": rpm,
        "map_inhg": map_inhg,
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
        ambient = RNG.uniform(15, 40)
        units.append(generate_unit(unit_id, "none", ambient))
        unit_id += 1
    for fault_mode in FAULT_MODES[1:]:
        for _ in range(n_faulty_per_mode):
            ambient = RNG.uniform(15, 40)
            units.append(generate_unit(unit_id, fault_mode, ambient))
            unit_id += 1
    return pd.concat(units, ignore_index=True)


def stream_live_telemetry(fault_mode: str = "none", interval_sec: float = 0.2):
    """
    Yields pre-computed engine packets at a fixed real-time cadence.

    This is a *replay* helper for local CLI testing (pre-computes one full
    mission using an isolated RNG, then drip-feeds it via blocking
    time.sleep) -- it is NOT safe to call directly from an async server.
    For a genuinely live, indefinitely-running, async-safe stream, use
    twin-core/simulator/live_stream.py's EngineUnitSimulator instead, which
    advances one cycle per call with no upfront full-mission computation
    and uses `await asyncio.sleep()` under FastAPI.
    """
    df_unit = generate_unit(unit_id=999, fault_mode=fault_mode, rng=_STREAM_RNG)
    for record in df_unit.to_dict(orient="records"):
        yield record
        time.sleep(interval_sec)


if __name__ == "__main__":
    import os

    # 1. Generate static CSV for model training
    out_path = os.path.join(os.path.dirname(__file__), "engine_telemetry_dataset.csv")
    df = generate_dataset()
    df.to_csv(out_path, index=False)
    print(f"Generated {df['unit_id'].nunique()} units, {len(df)} total rows -> {out_path}")

    # 2. Example of live streaming mode (local CLI demo only -- see docstring)
    print("\nStarting live stream test (Ctrl+C to stop)...")
    for packet in stream_live_telemetry(fault_mode="oil_degradation", interval_sec=0.5):
        print(f"RPM: {packet['rpm']:.0f} | MAP: {packet['map_inhg']:.1f} inHg | "
              f"Oil Press: {packet['oil_press_psi']:.1f} PSI")