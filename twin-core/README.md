# twin-core (Group 1)

**Owns:** A → Digital Twin Core Framework, B → Health Monitoring data path.

## Scope

- Ingest engine sensor data — for now, from `data/engine_data_generator.py`
  / `data/engine_telemetry_dataset.csv`; later, from a CAN bus / ECU-FADEC
  interface or a test rig.
- Maintain the "virtual engine" state: the latest synchronized reading per
  engine unit, matched against known operating limits.
- Emit a `telemetry_frame` (see `../docs/DATA_CONTRACT.md`) that
  `ml-models` and `dashboard` consume — via a local API, message queue, or
  simply a shared file/stream, whichever the team picks first.
- Physics-informed modelling: the relations in
  `data/engine_data_generator.py::_physics_informed_readings` (RPM →
  CHT/EGT/fuel flow/oil press/oil temp/vibration) are a starting point —
  refine them into an actual thermodynamic model of the Rotax 912-class
  engine as the twin's "physics layer".

## Layout

```
twin-core/
├── src/
│   ├── ingestion.py       # reads CSV/stream, yields raw telemetry rows
│   ├── state_estimator.py # physics-informed state + limit checking
│   └── api.py             # exposes telemetry_frame (FastAPI stub)
├── tests/
└── requirements.txt
```

## Quickstart

```bash
cd twin-core
pip install -r requirements.txt
python src/api.py
```

This serves `telemetry_frame`s from the sample dataset in `../data/` at
`http://localhost:8001/telemetry/latest`.

## Next steps for this group

1. Replace `ingestion.py`'s CSV replay with a real-time source (simulated
   CAN bus frames are a reasonable stand-in for a physical test rig).
2. Flesh out `state_estimator.py`'s physics model — this is the
   "thermodynamic behavior model + performance maps" the PS asks for.
3. Decide on the transport for `telemetry_frame` (REST polling, WebSocket
   push, or a message broker) and document it here once chosen.
