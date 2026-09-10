import os
import time

import numpy as np
import pandas as pd

# =============================================================================
# 1. Reproducibility
# =============================================================================

RNG = np.random.default_rng(42)


# =============================================================================
# 2. Baseline Engine Parameters
# =============================================================================
#
# Rotax 912-class reference values.
# These are simplified operating targets for synthetic data generation.
# They should not be treated as exact engine certification limits.
# =============================================================================

BASELINE = {
    # Engine speed
    "rpm_idle": 1400,
    "rpm_cruise": 5000,
    "rpm_max_cont": 5500,

    # Cylinder head temperature
    "cht_normal_c": 110,
    "cht_max_c": 150,

    # Exhaust gas temperature
    "egt_normal_c": 800,
    "egt_max_c": 900,

    # Oil pressure
    "oil_press_normal_psi": 58,
    "oil_press_min_psi": 22,
    "oil_press_max_psi": 72,

    # Oil temperature
    "oil_temp_normal_c": 100,
    "oil_temp_max_c": 120,

    # Fuel flow
    "fuel_flow_idle_lph": 4.0,
    "fuel_flow_cruise_lph": 15.0,

    # Vibration
    "vibration_normal_g": 0.5,

    # MAP
    "map_idle_inhg": 12.0,
    "map_max_inhg": 28.0,
}


# =============================================================================
# 3. Mission Profile
# =============================================================================
#
# phase, duration, target RPM
#
# Duration represents the number of generated telemetry samples.
# =============================================================================

MISSION_PROFILE = [
    ("idle", 30, BASELINE["rpm_idle"]),
    ("climb", 60, BASELINE["rpm_max_cont"]),
    ("cruise", 300, BASELINE["rpm_cruise"]),
    ("descent", 60, BASELINE["rpm_idle"] + 800),
    ("idle", 20, BASELINE["rpm_idle"]),
]


# =============================================================================
# 4. Supported Fault Modes
# =============================================================================

FAULT_MODES = [
    "none",
    "overheating",
    "misfire_vibration",
    "oil_degradation",
    "sensor_drift",
    "injector_fault",
]


# =============================================================================
# 5. Utility Functions
# =============================================================================

def _apply_thermal_lag(
    raw_temp: np.ndarray,
    alpha: float = 0.05,
) -> np.ndarray:
    """
    Simulate thermal inertia.

    A lower alpha means the temperature changes more slowly.
    This prevents CHT and oil temperature from reacting instantly to RPM.
    """

    lagged = np.empty_like(raw_temp)

    lagged[0] = raw_temp[0]

    for t in range(1, len(raw_temp)):
        lagged[t] = (
            alpha * raw_temp[t]
            + (1.0 - alpha) * lagged[t - 1]
        )

    return lagged


def _build_fault_progress(
    n_samples: int,
    fault_mode: str,
) -> np.ndarray:
    """
    Create a gradual fault-degradation curve.

    Healthy engine:
        progress = 0

    Faulty engine:
        fault begins between 30% and 50% of the mission and gradually
        increases toward 1.0.
    """

    if fault_mode == "none":
        return np.zeros(n_samples)

    onset = int(
        n_samples * RNG.uniform(0.30, 0.50)
    )

    progress = np.clip(
        (np.arange(n_samples) - onset)
        / max(1, n_samples - onset),
        0,
        1,
    )

    return progress


# =============================================================================
# 6. Base Physics-Informed Sensor Model
# =============================================================================

def _physics_informed_readings(
    rpm: np.ndarray,
    ambient_c: float = 25.0,
):
    """
    Derive dependent sensor values from RPM using simplified relationships.

    RPM influences:
        RPM → MAP
        RPM → fuel flow
        RPM → EGT
        RPM → CHT
        RPM → oil temperature
        RPM → oil pressure
        RPM → vibration
    """

    rpm_frac = (
        rpm - BASELINE["rpm_idle"]
    ) / (
        BASELINE["rpm_max_cont"]
        - BASELINE["rpm_idle"]
    )

    rpm_frac = np.clip(rpm_frac, 0.0, 1.0)

    # -------------------------------------------------------------------------
    # Cylinder Head Temperature
    # -------------------------------------------------------------------------

    raw_cht = (
        70.0
        + rpm_frac
        * (BASELINE["cht_normal_c"] - 70.0)
        + ambient_c * 0.15
    )

    cht = _apply_thermal_lag(
        raw_cht,
        alpha=0.03,
    )

    # -------------------------------------------------------------------------
    # Oil Temperature
    # -------------------------------------------------------------------------

    raw_oil_temp = (
        60.0
        + rpm_frac
        * (BASELINE["oil_temp_normal_c"] - 60.0)
        + ambient_c * 0.05
    )

    oil_temp = _apply_thermal_lag(
        raw_oil_temp,
        alpha=0.02,
    )

    # -------------------------------------------------------------------------
    # Exhaust Gas Temperature
    # -------------------------------------------------------------------------

    egt = (
        400.0
        + rpm_frac
        * (BASELINE["egt_normal_c"] - 400.0)
    )

    # -------------------------------------------------------------------------
    # Fuel Flow
    # -------------------------------------------------------------------------

    fuel_flow = (
        BASELINE["fuel_flow_idle_lph"]
        + rpm_frac
        * (
            BASELINE["fuel_flow_cruise_lph"]
            - BASELINE["fuel_flow_idle_lph"]
        )
    )

    # -------------------------------------------------------------------------
    # Oil Pressure
    # -------------------------------------------------------------------------

    oil_press = (
        BASELINE["oil_press_normal_psi"]
        - (1.0 - rpm_frac) * 10.0
    )

    # -------------------------------------------------------------------------
    # Vibration
    # -------------------------------------------------------------------------

    vibration = (
        BASELINE["vibration_normal_g"]
        * (0.5 + 0.5 * rpm_frac)
    )

    # -------------------------------------------------------------------------
    # Manifold Absolute Pressure
    # -------------------------------------------------------------------------

    map_inhg = (
        BASELINE["map_idle_inhg"]
        + rpm_frac
        * (
            BASELINE["map_max_inhg"]
            - BASELINE["map_idle_inhg"]
        )
    )

    return (
        cht,
        egt,
        fuel_flow,
        oil_press,
        oil_temp,
        vibration,
        map_inhg,
    )


# =============================================================================
# 7. Fault Injection
# =============================================================================

def _apply_fault(
    fault_mode: str,
    progress: np.ndarray,
    rpm: np.ndarray,
    cht: np.ndarray,
    egt: np.ndarray,
    fuel_flow: np.ndarray,
    oil_press: np.ndarray,
    oil_temp: np.ndarray,
    vibration: np.ndarray,
    map_inhg: np.ndarray,
):
    """
    Apply simplified fault signatures.

    Each fault modifies multiple related signals where appropriate.
    This is important because ML should learn a fault signature rather than
    simply learning one artificially corrupted sensor.
    """

    # =========================================================================
    # Overheating
    # =========================================================================

    if fault_mode == "overheating":

        # Cooling performance gradually decreases.
        cht = cht + progress * 45.0

        # Higher thermal load causes higher exhaust temperature.
        egt = egt + progress * 60.0

        # Oil temperature also increases because of thermal loading.
        oil_temp = oil_temp + progress * 15.0

    # =========================================================================
    # Misfire / Vibration Fault
    # =========================================================================

    elif fault_mode == "misfire_vibration":

        # Strong increase in vibration.
        vibration = (
            vibration
            + progress * 3.5
        )

        # At severe degradation, combustion becomes increasingly irregular.
        severe = progress > 0.70

        if np.any(severe):
            vibration[severe] += RNG.normal(
                0,
                0.4,
                severe.sum(),
            )

        # Misfire reduces effective combustion output.
        rpm_drop = (
            progress
            * RNG.uniform(50, 180)
        )

        rpm -= rpm_drop

        # Reduced combustion causes EGT instability.
        egt = (
            egt
            - progress * 35.0
        )

    # =========================================================================
    # Oil Degradation
    # =========================================================================

    elif fault_mode == "oil_degradation":

        # Increasing wear / lubrication degradation.
        oil_press = (
            oil_press
            - progress * 30.0
        )

        # Poorer lubrication produces higher oil temperature.
        oil_temp = (
            oil_temp
            + progress * 25.0
        )

        # Increased friction creates a small thermal load.
        cht = (
            cht
            + progress * 8.0
        )

    # =========================================================================
    # Sensor Drift
    # =========================================================================

    elif fault_mode == "sensor_drift":

        # Choose ONE sensor for the entire generated unit.
        #
        # This is intentionally deterministic for this generated unit,
        # rather than choosing a new sensor at every sample.
        drift_sensor = RNG.choice(
            ["egt", "cht", "oil_press"]
        )

        if drift_sensor == "egt":

            egt = (
                egt
                + progress * 80.0
            )

        elif drift_sensor == "cht":

            cht = (
                cht
                + progress * 30.0
            )

        elif drift_sensor == "oil_press":

            # Positive pressure drift.
            oil_press = (
                oil_press
                + progress * 20.0
            )

    # =========================================================================
    # Injector Fault
    # =========================================================================
    #
    # Intended fault chain:
    #
    # Injector degradation
    #        ↓
    # Abnormal fuel delivery
    #        ↓
    # EGT deviation
    #        ↓
    # Combustion instability
    #        ↓
    # RPM fluctuation
    #        ↓
    # Increased vibration
    #
    # =========================================================================

    elif fault_mode == "injector_fault":

        # ---------------------------------------------------------------------
        # 1. Abnormal fuel delivery
        # ---------------------------------------------------------------------
        #
        # A degraded injector causes progressively abnormal fuel delivery.
        #
        # The increase is deliberately moderate so that the signal remains
        # plausible instead of becoming an obviously artificial spike.
        # ---------------------------------------------------------------------

        fuel_flow = (
            fuel_flow
            + progress * 4.0
        )

        # ---------------------------------------------------------------------
        # 2. EGT deviation
        # ---------------------------------------------------------------------
        #
        # Richer / uneven combustion produces increased exhaust temperature
        # in this simplified fault model.
        # ---------------------------------------------------------------------

        egt = (
            egt
            + progress * 50.0
        )

        # ---------------------------------------------------------------------
        # 3. Combustion instability
        # ---------------------------------------------------------------------

        combustion_instability = (
            progress
            * 0.6
            * np.sin(
                np.arange(len(progress))
                * 0.7
            )
        )

        # Small EGT oscillation from unstable combustion.
        egt = (
            egt
            + combustion_instability * 20.0
        )

        # ---------------------------------------------------------------------
        # 4. RPM fluctuation
        # ---------------------------------------------------------------------

        rpm_variation = (
            progress
            * 120.0
            * np.sin(
                np.arange(len(progress))
                * 0.8
            )
        )

        rpm -= rpm_variation

        # ---------------------------------------------------------------------
        # 5. Increased vibration
        # ---------------------------------------------------------------------

        vibration = (
            vibration
            + progress * 1.2
            + np.abs(combustion_instability)
        )

        # ---------------------------------------------------------------------
        # 6. Small CHT increase
        # ---------------------------------------------------------------------

        cht = (
            cht
            + progress * 10.0
        )

    return (
        rpm,
        cht,
        egt,
        fuel_flow,
        oil_press,
        oil_temp,
        vibration,
        map_inhg,
    )


# =============================================================================
# 8. Generate One Engine Unit
# =============================================================================

def generate_unit(
    unit_id: int,
    fault_mode: str = "none",
    ambient_c: float = 25.0,
) -> pd.DataFrame:
    """
    Generate one complete mission time series for a single engine unit.
    """

    if fault_mode not in FAULT_MODES:
        raise ValueError(
            f"Unsupported fault_mode '{fault_mode}'. "
            f"Supported modes: {FAULT_MODES}"
        )

    # =========================================================================
    # Mission RPM Trace
    # =========================================================================

    rpm_trace = []
    phase_trace = []

    for phase, duration, rpm_target in MISSION_PROFILE:

        start_rpm = (
            rpm_trace[-1]
            if rpm_trace
            else BASELINE["rpm_idle"]
        )

        ramp = np.linspace(
            start_rpm,
            rpm_target,
            duration,
        )

        rpm_trace.extend(ramp)

        phase_trace.extend(
            [phase] * duration
        )

    rpm = np.array(
        rpm_trace,
        dtype=float,
    )

    n_cycles = len(rpm)

    # =========================================================================
    # Base RPM Sensor Noise
    # =========================================================================

    rpm += RNG.normal(
        0,
        15,
        n_cycles,
    )

    # =========================================================================
    # Base Physics-Informed Sensor Values
    # =========================================================================

    (
        cht,
        egt,
        fuel_flow,
        oil_press,
        oil_temp,
        vibration,
        map_inhg,
    ) = _physics_informed_readings(
        rpm,
        ambient_c,
    )

    # =========================================================================
    # Fault Progression
    # =========================================================================

    progress = _build_fault_progress(
        n_cycles,
        fault_mode,
    )

    # =========================================================================
    # Apply Fault
    # =========================================================================

    (
        rpm,
        cht,
        egt,
        fuel_flow,
        oil_press,
        oil_temp,
        vibration,
        map_inhg,
    ) = _apply_fault(
        fault_mode=fault_mode,
        progress=progress,
        rpm=rpm,
        cht=cht,
        egt=egt,
        fuel_flow=fuel_flow,
        oil_press=oil_press,
        oil_temp=oil_temp,
        vibration=vibration,
        map_inhg=map_inhg,
    )

    # =========================================================================
    # Sensor Noise
    # =========================================================================

    cht += RNG.normal(
        0,
        1.5,
        n_cycles,
    )

    egt += RNG.normal(
        0,
        5,
        n_cycles,
    )

    fuel_flow += RNG.normal(
        0,
        0.15,
        n_cycles,
    )

    oil_press += RNG.normal(
        0,
        1.0,
        n_cycles,
    )

    oil_temp += RNG.normal(
        0,
        1.0,
        n_cycles,
    )

    vibration += RNG.normal(
        0,
        0.05,
        n_cycles,
    )

    map_inhg += RNG.normal(
        0,
        0.2,
        n_cycles,
    )

    # =========================================================================
    # Safety Bounds
    # =========================================================================
    #
    # These prevent the synthetic generator from producing impossible negative
    # sensor values after fault injection and noise.
    # =========================================================================

    rpm = np.clip(
        rpm,
        800,
        6000,
    )

    fuel_flow = np.clip(
        fuel_flow,
        0,
        None,
    )

    oil_press = np.clip(
        oil_press,
        0,
        None,
    )

    oil_temp = np.clip(
        oil_temp,
        -20,
        180,
    )

    vibration = np.clip(
        vibration,
        0,
        None,
    )

    map_inhg = np.clip(
        map_inhg,
        5,
        35,
    )

    # =========================================================================
    # Remaining Useful Life
    # =========================================================================
    #
    # Healthy engines:
    #     RUL = 999
    #
    # Faulty engines:
    #     RUL decreases as the mission progresses.
    #
    # This is a synthetic target for ML experimentation.
    # =========================================================================

    if fault_mode != "none":

        rul = (
            n_cycles
            - np.arange(n_cycles)
        )

        rul = np.minimum(
            rul,
            200,
        )

    else:

        rul = np.full(
            n_cycles,
            999,
        )

    # =========================================================================
    # Build DataFrame
    # =========================================================================

    return pd.DataFrame(
        {
            "unit_id": unit_id,
            "cycle": np.arange(n_cycles),
            "phase": phase_trace,

            # Engine state
            "rpm": rpm,
            "map_inhg": map_inhg,

            # Thermal
            "cht_c": cht,
            "egt_c": egt,

            # Fuel
            "fuel_flow_lph": fuel_flow,

            # Lubrication
            "oil_press_psi": oil_press,
            "oil_temp_c": oil_temp,

            # Mechanical
            "vibration_g": vibration,

            # Environment
            "ambient_c": ambient_c,

            # Ground-truth labels
            "fault_mode": fault_mode,
            "RUL": rul,
        }
    )


# =============================================================================
# 9. Generate Full Training Dataset
# =============================================================================

def generate_dataset(
    n_healthy: int = 20,
    n_faulty_per_mode: int = 8,
) -> pd.DataFrame:
    """
    Generate a complete multi-unit dataset containing healthy and faulty
    engine missions.
    """

    units = []

    unit_id = 1

    # =========================================================================
    # Healthy Engines
    # =========================================================================

    for _ in range(n_healthy):

        ambient = RNG.uniform(
            15,
            40,
        )

        units.append(
            generate_unit(
                unit_id=unit_id,
                fault_mode="none",
                ambient_c=ambient,
            )
        )

        unit_id += 1

    # =========================================================================
    # Faulty Engines
    # =========================================================================

    for fault_mode in FAULT_MODES[1:]:

        for _ in range(n_faulty_per_mode):

            ambient = RNG.uniform(
                15,
                40,
            )

            units.append(
                generate_unit(
                    unit_id=unit_id,
                    fault_mode=fault_mode,
                    ambient_c=ambient,
                )
            )

            unit_id += 1

    # =========================================================================
    # Combine
    # =========================================================================

    return pd.concat(
        units,
        ignore_index=True,
    )


# =============================================================================
# 10. Live Telemetry Stream
# =============================================================================

def stream_live_telemetry(
    fault_mode: str = "none",
    interval_sec: float = 0.2,
):
    """
    Yield telemetry packets one sample at a time.

    This is intended for the RavenClaw Digital Twin.

    Example:

        for packet in stream_live_telemetry(
            fault_mode="injector_fault",
            interval_sec=0.2,
        ):
            print(packet)

    The same generated telemetry should be passed through the RavenClaw
    telemetry pipeline instead of creating a separate Digital Twin simulator.
    """

    df_unit = generate_unit(
        unit_id=999,
        fault_mode=fault_mode,
    )

    for record in df_unit.to_dict(
        orient="records"
    ):

        yield record

        time.sleep(
            interval_sec
        )


# =============================================================================
# 11. Script Entry Point
# =============================================================================

if __name__ == "__main__":

    # =========================================================================
    # Generate Static CSV
    # =========================================================================

    out_path = os.path.join(
        os.path.dirname(__file__),
        "engine_telemetry_dataset.csv",
    )

    df = generate_dataset()

    df.to_csv(
        out_path,
        index=False,
    )

    print(
        f"Generated "
        f"{df['unit_id'].nunique()} units, "
        f"{len(df)} total rows "
        f"-> {out_path}"
    )

    print("\nFault distribution:")

    print(
        df.groupby("fault_mode")
        .size()
        .to_string()
    )

    # =========================================================================
    # Live Stream Test
    # =========================================================================

    print(
        "\nStarting live stream test "
        "(Ctrl+C to stop)..."
    )

    try:

        for packet in stream_live_telemetry(
            fault_mode="injector_fault",
            interval_sec=0.5,
        ):

            print(
                f"RPM: {packet['rpm']:.0f} | "
                f"MAP: {packet['map_inhg']:.1f} inHg | "
                f"CHT: {packet['cht_c']:.1f} C | "
                f"EGT: {packet['egt_c']:.1f} C | "
                f"Fuel: {packet['fuel_flow_lph']:.2f} L/h | "
                f"Oil Press: {packet['oil_press_psi']:.1f} PSI | "
                f"Vibration: {packet['vibration_g']:.2f} g"
            )

    except KeyboardInterrupt:

        print(
            "\nLive stream stopped."
        )