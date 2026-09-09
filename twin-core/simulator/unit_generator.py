"""
Physics-informed single-unit engine simulator for twin-core.

This is the shared "engine model" that both dataset_generator.py (batch/
offline CSV generation) and live_stream.py (live cycle-by-cycle streaming)
build on, so the physics relations live in exactly one place instead of
being duplicated between here and data/engine_data_generator.py.

Baseline parameters are read from twin-core/config/engine_params.yaml
(Rotax 912-class) rather than hardcoded, so tuning the engine model means
editing one YAML file, not hunting through multiple .py files.
"""

from pathlib import Path
from typing import Dict

import numpy as np
import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "engine_params.yaml"


def _load_baseline() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


BASELINE = _load_baseline()

# One full mission: (phase_name, duration_cycles, target_rpm). 1 cycle ~= 1 Hz
# telemetry tick, matching the C-MAPSS-style convention used in data/.
MISSION_PROFILE = [
    ("idle", 30, BASELINE["rpm_idle"]),
    ("climb", 60, BASELINE["rpm_max_cont"]),
    ("cruise", 300, BASELINE["rpm_cruise"]),
    ("descent", 60, BASELINE["rpm_idle"] + 800),
    ("idle", 20, BASELINE["rpm_idle"]),
]


def physics_informed_readings(rpm: float, ambient_c: float = 25.0) -> Dict[str, float]:
    """
    Derive dependent sensor values from a single RPM reading, using the same
    simplified physics relations as data/engine_data_generator.py, ported to
    scalar inputs so a live stream can call this once per tick instead of
    only working over a pre-built array.
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


class EngineUnitSimulator:
    """
    Stateful, cycle-by-cycle simulator for one engine unit's mission.

    Call .step() once per tick to advance one cycle and get a fresh
    telemetry_frame-shaped dict (sensor noise included). Loops back to the
    start of the mission profile once it reaches the end, so a live demo
    can run indefinitely without restarting the process.
    """

    def __init__(self, unit_id: int, ambient_c: float = 25.0, seed: int = None):
        self.unit_id = unit_id
        self.ambient_c = ambient_c
        self.rng = np.random.default_rng(seed)
        self._phase_trace = []
        self._rpm_trace = self._build_rpm_trace()
        self.cycle = 0

    def _build_rpm_trace(self) -> np.ndarray:
        trace = []
        for phase, duration, rpm_target in MISSION_PROFILE:
            start_rpm = trace[-1] if trace else BASELINE["rpm_idle"]
            ramp = np.linspace(start_rpm, rpm_target, duration)
            trace.extend(ramp)
            self._phase_trace.extend([phase] * duration)
        return np.array(trace)

    @property
    def n_cycles(self) -> int:
        return len(self._rpm_trace)

    def step(self) -> dict:
        """Advance one cycle; returns a telemetry_frame-shaped dict."""
        idx = self.cycle % self.n_cycles
        rpm = float(self._rpm_trace[idx] + self.rng.normal(0, 15))
        sensors = physics_informed_readings(rpm, self.ambient_c)

        # sensor noise, same magnitudes as data/engine_data_generator.py
        sensors["cht_c"] += float(self.rng.normal(0, 1.5))
        sensors["egt_c"] += float(self.rng.normal(0, 5))
        sensors["oil_press_psi"] += float(self.rng.normal(0, 1.0))
        sensors["oil_temp_c"] += float(self.rng.normal(0, 1.0))
        sensors["vibration_g"] += float(self.rng.normal(0, 0.05))

        frame = {
            "unit_id": self.unit_id,
            "cycle": self.cycle,
            "phase": self._phase_trace[idx],
            "sensors": sensors,
        }
        self.cycle += 1
        return frame

    def reset(self) -> None:
        """Restart the mission from cycle 0 (e.g. for a fresh demo run)."""
        self.cycle = 0


if __name__ == "__main__":
    sim = EngineUnitSimulator(unit_id=1, ambient_c=28.0, seed=1)
    for _ in range(5):
        print(sim.step())