"""
Real-time telemetry push for twin-core.

api.py's /telemetry/latest only advances when someone GETs it — fine as a
placeholder, but it means the dashboard has to poll in a loop, and there's
no way to demo "inject a fault and watch it get caught live" without a
control channel. This gives you both:

  - WebSocket push of one telemetry_frame per second (or --interval),
    so dashboard just opens a socket and renders whatever arrives.
  - POST /inject_fault to kick off a live-developing fault mid-stream,
    using fault_injector.apply_fault instead of relying on the CSV's
    pre-baked fault_mode column.

Run with: python src/live_stream.py
Then:     ws://localhost:8002/ws/telemetry?unit_id=1
          curl -X POST localhost:8002/inject_fault -d '{"fault_mode": "overheating"}' \\
               -H "Content-Type: application/json"
"""

import asyncio
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from fault_injector import FAULT_MODES, apply_fault, reset_drift_state
from ingestion import csv_stream
from state_estimator import breach_flags, to_telemetry_frame

app = FastAPI(title="twin-core live stream")

PUSH_INTERVAL_SECONDS = 1.0
FAULT_RAMP_CYCLES = 60  # cycles from injection to full-severity (progress=1.0)


class _FaultState:
    """Tracks an operator-triggered live fault, if any, for the demo."""

    def __init__(self):
        self.active_mode: Optional[str] = None
        self.cycles_since_injection: int = 0

    def trigger(self, fault_mode: str) -> None:
        if fault_mode not in FAULT_MODES:
            raise ValueError(f"unknown fault_mode: {fault_mode!r}")
        reset_drift_state()
        self.active_mode = None if fault_mode == "none" else fault_mode
        self.cycles_since_injection = 0

    def progress(self) -> float:
        return min(self.cycles_since_injection / FAULT_RAMP_CYCLES, 1.0)

    def tick(self) -> None:
        if self.active_mode is not None:
            self.cycles_since_injection += 1


_fault_state = _FaultState()


class InjectFaultRequest(BaseModel):
    fault_mode: str  # one of FAULT_MODES; "none" clears an active fault


@app.post("/inject_fault")
def inject_fault(req: InjectFaultRequest):
    """Trigger (or clear) a live fault for the next connected stream(s)."""
    _fault_state.trigger(req.fault_mode)
    return {"active_mode": _fault_state.active_mode}


@app.websocket("/ws/telemetry")
async def ws_telemetry(websocket: WebSocket, unit_id: int = 1):
    """Push one telemetry_frame per PUSH_INTERVAL_SECONDS until disconnected."""
    await websocket.accept()
    stream = csv_stream(unit_id=unit_id)
    try:
        for row in stream:
            frame = to_telemetry_frame(row)

            if _fault_state.active_mode is not None:
                frame["sensors"] = apply_fault(
                    _fault_state.active_mode, _fault_state.progress(), frame["sensors"]
                )
                frame["injected_fault"] = {
                    "mode": _fault_state.active_mode,
                    "progress": round(_fault_state.progress(), 3),
                }
                _fault_state.tick()

            frame["breach_flags"] = breach_flags(frame)
            await websocket.send_json(frame)
            await asyncio.sleep(PUSH_INTERVAL_SECONDS)
    except WebSocketDisconnect:
        pass  # client closed the tab — nothing to clean up


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
