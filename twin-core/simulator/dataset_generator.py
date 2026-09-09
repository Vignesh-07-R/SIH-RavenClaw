"""
Batch dataset generation for twin-core.

Built on the same physics model and fault relations used for live streaming
(unit_generator.py and ../models/fault_injector.py), so offline training
data and live-demo behavior can't drift apart from each other.

This is twin-core's own version of data/engine_data_generator.py's
generate_dataset() -- kept here so ml-models can regenerate training data
against twin-core's canonical physics model as it evolves. If both
generators are still needed once the team settles on one, the simplest
fix is making data/engine_data_generator.py a thin wrapper that calls this
instead of duplicating the physics a second time.

Run with: python dataset_generator.py
Writes:   ../data/engine_telemetry_dataset.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from fault_injector import FAULT_MODES, apply_fault, reset_drift_state  # noqa: E402

from unit_generator import EngineUnitSimulator  # noqa: E402

RUL_CAP = 200


def generate_unit_dataframe(unit_id: int, fault_mode: str = "none",
                             ambient_c: float = 25.0, seed: int = None) -> pd.DataFrame:
    """
    Run one unit through a full mission (one pass through the profile) and
    return a DataFrame shaped like data/engine_telemetry_dataset.csv.
    """
    reset_drift_state()  # so sensor_drift's coin-flip doesn't reuse the last unit's direction
    sim = EngineUnitSimulator(unit_id=unit_id, ambient_c=ambient_c, seed=seed)
    n_cycles = sim.n_cycles
    rng = np.random.default_rng(seed)

    # fault starts partway through the mission, same convention as data/
    onset = 0 if fault_mode == "none" else int(n_cycles * rng.uniform(0.3, 0.5))

    rows = []
    for _ in range(n_cycles):
        frame = sim.step()
        progress = 0.0
        if fault_mode != "none" and frame["cycle"] >= onset:
            progress = min((frame["cycle"] - onset) / max(n_cycles - onset, 1), 1.0)
        sensors = apply_fault(fault_mode, progress, frame["sensors"])

        rul = RUL_CAP if fault_mode == "none" else min(n_cycles - frame["cycle"], RUL_CAP)
        rows.append({
            "unit_id": unit_id,
            "cycle": frame["cycle"],
            "phase": frame["phase"],
            **sensors,
            "fault_mode": fault_mode,
            "RUL": rul,
        })
    return pd.DataFrame(rows)


def generate_dataset(n_healthy: int = 20, n_faulty_per_mode: int = 8, seed: int = 42) -> pd.DataFrame:
    """Generate a full multi-unit dataset, mixing healthy and faulty units."""
    rng = np.random.default_rng(seed)
    units, unit_id = [], 1

    for _ in range(n_healthy):
        ambient = rng.uniform(15, 40)
        units.append(generate_unit_dataframe(unit_id, "none", ambient, seed=unit_id))
        unit_id += 1

    for fault_mode in FAULT_MODES[1:]:
        for _ in range(n_faulty_per_mode):
            ambient = rng.uniform(15, 40)
            units.append(generate_unit_dataframe(unit_id, fault_mode, ambient, seed=unit_id))
            unit_id += 1

    return pd.concat(units, ignore_index=True)


if __name__ == "__main__":
    df = generate_dataset()
    out_path = Path(__file__).resolve().parents[1] / "data" / "engine_telemetry_dataset.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Generated {df['unit_id'].nunique()} units, {len(df)} rows -> {out_path}")
    print(df.groupby("fault_mode")["unit_id"].nunique())