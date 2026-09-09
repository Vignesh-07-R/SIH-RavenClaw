"""
Stateful, cycle-by-cycle engine unit simulator for twin-core.

This is the orchestration layer: it owns *state* (which cycle a unit is on,
its RNG) and calls into the domain models in ../models/ for the physics
(physics_engine.physics_informed_readings) and the mission shape
(mission_profile.build_rpm_and_phase_trace). Keeping those as plain,
stateless functions in models/ means ml-models or a notebook can reuse the
physics without instantiating a simulator.

Used by:
  - dataset_generator.py  -- runs .step() n_cycles times per unit, batched
  - live_stream.py        -- runs .step() once per second, indefinitely
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from mission_profile import build_rpm_and_phase_trace  # noqa: E402
from physics_engine import physics_informed_readings  # noqa: E402


class EngineUnitSimulator:
    """
    Call .step() once per tick to advance one cycle and get a fresh
    telemetry_frame-shaped dict (sensor noise included). Loops back to the
    start of the mission profile once it reaches the end, so a live demo
    can run indefinitely without restarting the process.
    """

    def __init__(self, unit_id: int, ambient_c: float = 25.0, seed: int = None, profile=None):
        self.unit_id = unit_id
        self.ambient_c = ambient_c
        self.rng = np.random.default_rng(seed)
        self._rpm_trace, self._phase_trace = build_rpm_and_phase_trace(profile)
        self.cycle = 0

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