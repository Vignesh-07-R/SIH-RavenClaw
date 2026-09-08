"""
Minimal API so ml-models and dashboard have something to point at from
day one. Run with: python src/api.py
Then: curl http://localhost:8001/telemetry/latest
"""

from fastapi import FastAPI
import uvicorn

from ingestion import csv_stream
from state_estimator import to_telemetry_frame

app = FastAPI(title="twin-core")

# In-memory cursor over the sample dataset, standing in for a live feed.
_stream = csv_stream(unit_id=1)
_latest = None


@app.get("/telemetry/latest")
def latest():
    """Return the next telemetry_frame in the replay (advances each call)."""
    global _latest
    try:
        row = next(_stream)
        _latest = to_telemetry_frame(row)
    except StopIteration:
        pass  # keep returning the last frame once the replay ends
    return _latest


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
