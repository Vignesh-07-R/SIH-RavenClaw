# dashboard (Group 3)

**Owns:** F → Visualization Dashboard, and the operator-facing side of E →
Simulation & Replay.

## Scope

- Real-time engine health status (from `twin-core`'s `telemetry_frame`)
- Fault alerts and maintenance advisory (from `ml-models`' `health_report`)
- Engine efficiency trends over a mission
- Mission-wise health reports / post-flight replay view

## Layout

```
dashboard/
├── src/
│   └── app.py       # Streamlit app polling twin-core + ml-models
├── assets/          # logos, icons, static assets
└── requirements.txt
```

Built with **Streamlit** for a fast prototype — swap for React/Next.js
later if a production-grade HMI is needed; the data sources
(`telemetry_frame` / `health_report` over HTTP) don't change either way.

## Quickstart

Run `twin-core`'s API and `ml-models`' API first (see their READMEs), then:

```bash
cd dashboard
pip install -r requirements.txt
streamlit run src/app.py
```

Opens at `http://localhost:8501`.

## Next steps for this group

1. Wire up live polling against `twin-core` (`:8001`) and `ml-models`
   (`:8002`) — the stub already does this for a single engine unit.
2. Add a mission-replay view: load a full unit's history from
   `../data/engine_telemetry_dataset.csv` and scrub through it cycle by
   cycle (this doubles as your "post-flight analysis" deliverable).
3. Add a multi-unit fleet view once more than one unit needs monitoring
   simultaneously.
