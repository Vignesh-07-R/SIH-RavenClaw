"""
Mission profile for twin-core.

A "mission" is one idle -> climb -> cruise -> descent -> idle run, the
same shape every unit flies in data/engine_data_generator.py and
docs/DATA_CONTRACT.md's `phase` field. Keeping it in its own module (not
buried inside the simulator) means ml-models or dashboard can import the
phase list without pulling in simulation state, and a future "hot-weather
mission" or "high-altitude endurance mission" (both named explicitly in
the PS section E) is just a new profile added here.
"""

from typing import List, Tuple

import numpy as np

from physics_engine import BASELINE

# (phase_name, duration_cycles, target_rpm). 1 cycle ~= 1 Hz telemetry tick,
# matching the C-MAPSS-style per-cycle convention used across the project.
MISSION_PROFILE: List[Tuple[str, int, float]] = [
    ("idle", 30, BASELINE["rpm_idle"]),
    ("climb", 60, BASELINE["rpm_max_cont"]),
    ("cruise", 300, BASELINE["rpm_cruise"]),
    ("descent", 60, BASELINE["rpm_idle"] + 800),
    ("idle", 20, BASELINE["rpm_idle"]),
]

# PS section E also asks for rapid-throttle-transition and endurance-mission
# simulation. This is a placeholder shape for that -- fill in real
# duration/rpm values once the team decides what "rapid" means in cycles.
RAPID_THROTTLE_PROFILE: List[Tuple[str, int, float]] = [
    ("idle", 20, BASELINE["rpm_idle"]),
    ("climb", 10, BASELINE["rpm_max_cont"]),
    ("idle", 10, BASELINE["rpm_idle"]),
    ("climb", 10, BASELINE["rpm_max_cont"]),
    ("cruise", 60, BASELINE["rpm_cruise"]),
]


def build_rpm_and_phase_trace(profile: List[Tuple[str, int, float]] = None):
    """
    Expand a mission profile into a per-cycle RPM array and phase list,
    ramping smoothly between each phase's target RPM instead of stepping.

    Returns (rpm_trace: np.ndarray, phase_trace: list[str]), same length.
    """
    profile = profile or MISSION_PROFILE
    rpm_trace: List[float] = []
    phase_trace: List[str] = []
    for phase, duration, rpm_target in profile:
        start_rpm = rpm_trace[-1] if rpm_trace else BASELINE["rpm_idle"]
        ramp = np.linspace(start_rpm, rpm_target, duration)
        rpm_trace.extend(ramp)
        phase_trace.extend([phase] * duration)
    return np.array(rpm_trace), phase_trace