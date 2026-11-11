"""Flask ingest API: receives windows from the ESP32 and returns an assessment.

POST /ingest  {"device_id": "...", "samples": [{"vibration":..,"temperature":..,...}, ...]}
GET  /latest  most recent assessment + samples
GET  /health  liveness check

Run:  python -m api.ingest_server
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
from flask import Flask, jsonify, request

from src.config import MODEL_PATH, SENSORS, WINDOW_SIZE
from src.predict import PredictiveMaintenanceEngine
from src.train import train

app = Flask(__name__)
_state: dict = {"latest": None}
_engine: PredictiveMaintenanceEngine | None = None


def get_engine() -> PredictiveMaintenanceEngine:
    global _engine
    if _engine is None:
        if not MODEL_PATH.exists():
            train(400, verbose=False)
        _engine = PredictiveMaintenanceEngine()
    return _engine


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/ingest")
def ingest():
    payload = request.get_json(silent=True) or {}
    samples = payload.get("samples", [])
    if len(samples) < 8:
        return jsonify(error="need at least 8 samples"), 400
    df = pd.DataFrame(samples)
    if not any(s in df.columns for s in SENSORS):
        return jsonify(error=f"samples must contain sensor fields: {SENSORS}"), 400
    assessment = get_engine().assess(df.tail(WINDOW_SIZE))
    _state["latest"] = {
        "device_id": payload.get("device_id", "unknown"),
        "received_at": datetime.now().isoformat(timespec="seconds"),
        "assessment": assessment,
        "samples": df.tail(WINDOW_SIZE).to_dict(orient="records"),
    }
    return jsonify(assessment)


@app.get("/latest")
def latest():
    if _state["latest"] is None:
        return jsonify(error="no data yet"), 404
    return jsonify(_state["latest"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
