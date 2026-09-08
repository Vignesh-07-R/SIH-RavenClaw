"""
Turns a raw telemetry row into the shared `telemetry_frame` shape defined
in docs/DATA_CONTRACT.md. This is the "virtual engine model" layer: today
it just repackages sensor readings against known limits, but this is where
a real thermodynamic / performance-map model should live.
"""

from datetime import datetime, timezone

# Rotax 912-class operating limits (see engine_data_generator.py::BASELINE).
# Kept here so twin-core can flag breaches without ml-models or dashboard
# needing their own copy of engine specs.
LIMITS = {
    "cht_max_c": 150,
    "egt_max_c": 900,
    "oil_press_min_psi": 22,
    "oil_press_max_psi": 72,
    "oil_temp_max_c": 120,
}


def to_telemetry_frame(row: dict) -> dict:
    """Convert one raw CSV row into a telemetry_frame dict (see DATA_CONTRACT.md)."""
    return {
        "unit_id": int(row["unit_id"]),
        "cycle": int(row["cycle"]),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": row["phase"],
        "sensors": {
            "rpm": float(row["rpm"]),
            "cht_c": float(row["cht_c"]),
            "egt_c": float(row["egt_c"]),
            "fuel_flow_lph": float(row["fuel_flow_lph"]),
            "oil_press_psi": float(row["oil_press_psi"]),
            "oil_temp_c": float(row["oil_temp_c"]),
            "vibration_g": float(row["vibration_g"]),
            "ambient_c": float(row["ambient_c"]),
        },
        "limits": LIMITS,
    }


def breach_flags(frame: dict) -> dict:
    """Simple threshold check — the 'conventional' baseline ml-models should beat."""
    s, l = frame["sensors"], frame["limits"]
    return {
        "cht_over": s["cht_c"] > l["cht_max_c"],
        "egt_over": s["egt_c"] > l["egt_max_c"],
        "oil_press_out_of_range": not (l["oil_press_min_psi"] <= s["oil_press_psi"] <= l["oil_press_max_psi"]),
        "oil_temp_over": s["oil_temp_c"] > l["oil_temp_max_c"],
    }


if __name__ == "__main__":
    from ingestion import csv_stream

    for i, row in enumerate(csv_stream(unit_id=1)):
        frame = to_telemetry_frame(row)
        print(frame, breach_flags(frame))
        if i >= 2:
            break
