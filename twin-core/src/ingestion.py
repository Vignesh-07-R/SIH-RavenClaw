"""
Telemetry ingestion for twin-core.

For now this replays the synthetic dataset row-by-row, standing in for a
real-time CAN bus / ECU-FADEC feed. Swap `csv_stream()` for a real
subscriber (e.g. python-can) without changing anything downstream, since
everything downstream only depends on the `telemetry_frame` dict shape.
"""

from pathlib import Path
from typing import Iterator

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "engine_telemetry_dataset.csv"


def csv_stream(unit_id: int | None = None) -> Iterator[dict]:
    """
    Yield raw telemetry rows as dicts, one per cycle, in order.

    Args:
        unit_id: if given, stream only that engine unit's mission;
                 otherwise stream every unit in `unit_id` order.
    """
    df = pd.read_csv(DATA_PATH)
    if unit_id is not None:
        df = df[df["unit_id"] == unit_id]

    for _, row in df.sort_values(["unit_id", "cycle"]).iterrows():
        yield row.to_dict()


if __name__ == "__main__":
    # quick smoke test: print the first 3 rows of unit 1
    for i, frame in enumerate(csv_stream(unit_id=1)):
        print(frame)
        if i >= 2:
            break
