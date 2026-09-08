# Shared synthetic dataset

No real MALE UAV piston-engine telemetry is public, so `engine_data_generator.py`
simulates plausible data instead of inventing numbers:

1. **Baseline operating parameters** → Rotax 912-class engine specs (the
   engine family actually used in several MALE-class UAVs).
2. **Dataset shape / RUL labeling convention** → NASA C-MAPSS turbofan
   degradation benchmark (`unit_id`, `cycle`, settings, sensors, `RUL`).
   Only the *structure* is reused — C-MAPSS is a jet engine, not a piston
   engine.

Field-by-field schema: see [`../docs/DATA_CONTRACT.md`](../docs/DATA_CONTRACT.md).

## Files

- `engine_data_generator.py` — regenerate or resize the dataset:
  ```bash
  python engine_data_generator.py
  ```
  Edit `generate_dataset(n_healthy=..., n_faulty_per_mode=...)` at the
  bottom of the file to change dataset size. Uses a fixed random seed
  (`np.random.default_rng(42)`) so runs are reproducible.
- `engine_telemetry_dataset.csv` — a pre-generated sample (52 units, one
  mission each: idle → climb → cruise → descent → idle) so groups 2 and 3
  can start immediately.

## Fault modes simulated

| Fault | Effect |
|---|---|
| `overheating` | CHT/EGT trend upward past normal limits |
| `misfire_vibration` | Vibration rises with added noise bursts |
| `oil_degradation` | Oil pressure drops, oil temp rises |
| `sensor_drift` | One sensor (EGT) biases high, decoupled from real engine state |

Each faulty unit's fault onsets partway through the mission (30–50% in)
and progresses to a simulated failure point, giving a run-to-failure
trajectory for RUL training — same convention as C-MAPSS.
