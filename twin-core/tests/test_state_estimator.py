"""
This originally tested a `state_estimator.py` module that never actually
existed in the repo (to_telemetry_frame / breach_flags). That job is now
split across:
  - simulator/unit_generator.py's EngineUnitSimulator.step() -- builds the
    telemetry_frame shape directly, cycle by cycle
  - models/physics_engine.py's breach_flags() -- limit checking

breach_flags() has its own coverage in test_physics_engine.py. This file
now checks the frame shape .step() produces, which is what
test_to_telemetry_frame_shape() used to check.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulator"))

from unit_generator import EngineUnitSimulator  # noqa: E402


def test_step_produces_telemetry_frame_shape():
    sim = EngineUnitSimulator(unit_id=1, ambient_c=25.0, seed=0)
    frame = sim.step()
    assert frame["unit_id"] == 1
    assert frame["cycle"] == 0
    assert isinstance(frame["phase"], str)
    assert set(frame["sensors"]) == {
        "rpm", "cht_c", "egt_c", "fuel_flow_lph",
        "oil_press_psi", "oil_temp_c", "vibration_g", "ambient_c",
    }


def test_step_advances_cycle_each_call():
    sim = EngineUnitSimulator(unit_id=1, ambient_c=25.0, seed=0)
    first = sim.step()
    second = sim.step()
    assert second["cycle"] == first["cycle"] + 1


def test_step_wraps_around_after_full_mission():
    sim = EngineUnitSimulator(unit_id=1, ambient_c=25.0, seed=0)
    for _ in range(sim.n_cycles):
        sim.step()
    wrapped = sim.step()
    assert wrapped["cycle"] == sim.n_cycles  # cycle count keeps rising...
    # ...but the underlying trace index has looped back to the start
    assert wrapped["phase"] == sim._phase_trace[0]


def test_reset_returns_to_cycle_zero():
    sim = EngineUnitSimulator(unit_id=1, ambient_c=25.0, seed=0)
    sim.step()
    sim.step()
    sim.reset()
    assert sim.cycle == 0