"""Inference engine: health assessment, failure-risk prediction and
maintenance decision support for one window of sensor readings."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from .config import MODEL_PATH, RISK_LOW_MAX, RISK_MEDIUM_MAX
from .features import extract_features
from .preprocessing import clean_window

FAULT_ACTIONS = {
    "bearing_wear": "Inspect and lubricate bearings; replace if worn; check shaft alignment.",
    "imbalance": "Check rotor balance, coupling and mounting bolts; re-balance if needed.",
    "overheating": "Check cooling fan, ventilation path and winding temperature; clean heat sinks.",
    "overload": "Reduce mechanical load; inspect driven equipment for jamming or friction.",
    "voltage_fault": "Inspect supply voltage, phase balance and terminal connections.",
}


def risk_level(score: float) -> str:
    if score < RISK_LOW_MAX:
        return "Low"
    return "Medium" if score < RISK_MEDIUM_MAX else "High"


def maintenance_decision(state: str, level: str, fault: str | None, anomaly: bool) -> dict:
    """Maintenance Decision Support: recommendation, priority, corrective action."""
    if state == "CRITICAL":
        return {"priority": "Urgent",
                "recommendation": "Schedule immediate inspection; consider stopping the motor.",
                "corrective_action": FAULT_ACTIONS.get(fault, "Perform full motor inspection."),
                "alert": True}
    if state == "WARNING":
        prio = "High" if level == "High" else "Medium"
        return {"priority": prio,
                "recommendation": "Plan maintenance in the next available window and monitor closely.",
                "corrective_action": FAULT_ACTIONS.get(fault, "Inspect motor for early-stage faults."),
                "alert": True}
    if anomaly:
        return {"priority": "Low",
                "recommendation": "Unusual pattern detected but within healthy class; keep monitoring.",
                "corrective_action": "No action yet; review the top indicators below.",
                "alert": False}
    return {"priority": "None", "recommendation": "Motor operating normally.",
            "corrective_action": "Continue routine monitoring.", "alert": False}


class PredictiveMaintenanceEngine:
    def __init__(self, model_path=MODEL_PATH):
        self.bundle = joblib.load(model_path)
        self.cols = self.bundle["feature_columns"]

    @property
    def metrics(self) -> dict:
        return self.bundle["metrics"]

    def assess(self, window: pd.DataFrame) -> dict:
        """window: DataFrame with sensor columns (and optional timestamp)."""
        b = self.bundle
        feats = extract_features(clean_window(window))
        x = pd.DataFrame([feats]).reindex(columns=self.cols)
        x = x.fillna(pd.Series(b["medians"]))

        proba = b["health_model"].predict_proba(x)[0]
        probs = dict(zip(b["health_model"].classes_, proba))
        state = max(probs, key=probs.get)
        score = float(probs.get("WARNING", 0) * 0.5 + probs.get("CRITICAL", 0))
        level = risk_level(score)

        xs = b["anomaly_scaler"].transform(x)
        anomaly = bool(b["anomaly_model"].predict(xs)[0] == -1)
        anomaly_score = float(-b["anomaly_model"].score_samples(xs)[0])

        fault = str(b["fault_model"].predict(x)[0]) if state != "HEALTHY" else None

        z = {c: (x[c].iloc[0] - b["healthy_mean"][c]) / (b["healthy_std"][c] or 1.0) for c in self.cols}
        top = sorted(z.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]

        result = {
            "health_state": state,
            "probabilities": {k: round(float(v), 4) for k, v in probs.items()},
            "risk_score": round(score, 4),
            "risk_level": level,
            "anomaly": anomaly,
            "anomaly_score": round(anomaly_score, 4),
            "fault_type": fault,
            "top_indicators": [{"feature": k, "z_score": round(float(v), 2)} for k, v in top],
        }
        result.update(maintenance_decision(state, level, fault, anomaly))
        return result
