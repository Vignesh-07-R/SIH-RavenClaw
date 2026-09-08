# SIH26054 — Digital Twin for MALE UAV Aero Piston Engines

AI-enabled real-time digital twin for health monitoring, fault prediction,
and mission reliability of aero piston engines used in MALE UAVs.

Problem statement: **SIH26054** (DRDO, Robotics & Drones theme).
Full text: [`docs/PROBLEM_STATEMENT.md`](docs/PROBLEM_STATEMENT.md).

## Team split (3 groups, 3 folders)

| Folder | Group | Owns |
|---|---|---|
| [`twin-core/`](twin-core/) | Group 1 | Data ingestion, physics-informed virtual engine model, real-time state sync |
| [`ml-models/`](ml-models/) | Group 2 | Anomaly detection, RUL estimation, fault classification |
| [`dashboard/`](dashboard/) | Group 3 | Operator/maintenance HMI, alerts, mission-wise reports |

Each folder is a **self-contained module** with its own `README.md`,
`requirements.txt`, `src/`, and `tests/`. Groups should only need to touch
their own folder plus `docs/DATA_CONTRACT.md` if a schema change is needed
(and that change must be agreed by all three groups, since it's the shared
interface — see below).

## How the pieces fit together

```
 Engine sensors / synthetic generator
              │
              ▼
   ┌─────────────────┐   telemetry_frame (JSON/CSV, see DATA_CONTRACT.md)
   │   twin-core      │ ─────────────────────────────┐
   │  (Group 1)       │                               │
   └─────────────────┘                               ▼
                                              ┌─────────────────┐
                                              │   ml-models      │
                                              │  (Group 2)       │
                                              └─────────────────┘
                                                       │
                                    health_report (JSON, see DATA_CONTRACT.md)
                                                       ▼
                                              ┌─────────────────┐
                                              │   dashboard      │
                                              │  (Group 3)       │
                                              └─────────────────┘
```

- **twin-core** turns raw/simulated sensor streams into a synchronized engine
  state (`telemetry_frame`).
- **ml-models** consumes `telemetry_frame`s and produces a `health_report`
  (anomaly flags, RUL estimate, fault probabilities).
- **dashboard** consumes both `telemetry_frame` and `health_report` and
  renders them for operators.

Because the interface is a plain JSON/CSV schema (not shared code), each
group can build, test, and demo independently before wiring everything
together — that's what avoids merge collisions.

## Shared data

- [`data/engine_data_generator.py`](data/engine_data_generator.py) — synthetic
  telemetry generator (Rotax 912-class baseline physics + C-MAPSS-style
  run-to-failure labeling). Everyone uses this until/unless real test-rig
  data becomes available.
- [`data/engine_telemetry_dataset.csv`](data/engine_telemetry_dataset.csv) —
  a pre-generated sample dataset (52 units) so groups 2 and 3 can start
  immediately without running the generator themselves.
- [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md) — the exact schema for
  every field, plus the `telemetry_frame` / `health_report` JSON shapes used
  at runtime.

## Getting started

```bash
git clone <your-repo-url>
cd sih26054-engine-dt

# each group works inside its own folder, e.g.:
cd twin-core && pip install -r requirements.txt
```

See each folder's README for module-specific setup and next steps.

## Repo etiquette (so 3 groups don't collide)

1. **Don't edit another group's folder.** If you need a change there, open
   an issue/PR instead of editing directly.
2. **Treat `docs/DATA_CONTRACT.md` as an API contract.** Changing a field
   name or type there breaks the other two groups — discuss before changing.
3. **One branch per group**, e.g. `twin-core-dev`, `ml-models-dev`,
   `dashboard-dev`; merge to `main` via PR.
4. Keep large binary/data files out of ad-hoc commits — put generated
   datasets in `data/` and add anything huge to `.gitignore`.
