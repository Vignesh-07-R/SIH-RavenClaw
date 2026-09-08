# Data Contract

This is the shared interface between `twin-core`, `ml-models`, and
`dashboard`. All three groups should treat this file as the source of truth
for field names and types. **Propose changes via PR, not silent edits.**

## 1. Raw telemetry row (CSV / DataFrame)

Produced by `data/engine_data_generator.py`, or in production by
`twin-core`'s ingestion layer reading real ECU/FADEC/CAN bus data.

| Field | Type | Units | Notes |
|---|---|---|---|
| `unit_id` | int | – | one engine "mission" / run-to-failure trajectory |
| `cycle` | int | sample index | 1 sample ≈ 1 Hz telemetry tick |
| `phase` | str | – | `idle`, `climb`, `cruise`, `descent` |
| `rpm` | float | rev/min | |
| `cht_c` | float | °C | Cylinder Head Temperature |
| `egt_c` | float | °C | Exhaust Gas Temperature |
| `fuel_flow_lph` | float | L/h | |
| `oil_press_psi` | float | psi | |
| `oil_temp_c` | float | °C | |
| `vibration_g` | float | g | |
| `ambient_c` | float | °C | environmental condition for that mission |
| `fault_mode` | str | – | ground-truth label: `none`, `overheating`, `misfire_vibration`, `oil_degradation`, `sensor_drift` (used for training/eval only — not available at inference time) |
| `RUL` | int | cycles | ground-truth Remaining Useful Life, capped at 200 (used for training/eval only) |

## 2. `telemetry_frame` — twin-core → ml-models / dashboard

One synchronized snapshot of engine state, emitted per cycle. JSON shape:

```json
{
  "unit_id": 1,
  "cycle": 142,
  "timestamp": "2026-09-08T10:15:32Z",
  "phase": "cruise",
  "sensors": {
    "rpm": 5001.2,
    "cht_c": 108.4,
    "egt_c": 795.1,
    "fuel_flow_lph": 14.8,
    "oil_press_psi": 57.9,
    "oil_temp_c": 99.6,
    "vibration_g": 0.51,
    "ambient_c": 27.3
  },
  "limits": {
    "cht_max_c": 150,
    "egt_max_c": 900,
    "oil_press_min_psi": 22,
    "oil_press_max_psi": 72,
    "oil_temp_max_c": 120
  }
}
```

`limits` lets the dashboard and ml-models flag threshold breaches without
hardcoding engine specs in three places.

## 3. `health_report` — ml-models → dashboard

Emitted whenever ml-models scores a `telemetry_frame` (or a batch, for
post-flight replay). JSON shape:

```json
{
  "unit_id": 1,
  "cycle": 142,
  "timestamp": "2026-09-08T10:15:32Z",
  "anomaly": {
    "is_anomalous": true,
    "score": 0.83
  },
  "rul_estimate_cycles": 58,
  "fault_probabilities": {
    "none": 0.05,
    "overheating": 0.72,
    "misfire_vibration": 0.03,
    "oil_degradation": 0.15,
    "sensor_drift": 0.05
  },
  "advisory": "CHT trending toward limit — recommend ground inspection within 60 cycles."
}
```

## 4. Versioning

If a field must change (rename, new unit, new fault mode, etc.):

1. Bump a `schema_version` string at the top of this file.
2. Update this doc in the same PR as the code change.
3. Ping the other two groups before merging — they consume this shape.
