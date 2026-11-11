"""Central configuration for the predictive-maintenance project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "pm_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

# Condition-monitoring sensors (units are used only for display)
SENSORS = ["vibration", "temperature", "current", "voltage", "speed", "pressure"]
UNITS = {
    "vibration": "mm/s RMS",
    "temperature": "°C",
    "current": "A",
    "voltage": "V",
    "speed": "RPM",
    "pressure": "bar",
}

WINDOW_SIZE = 64          # readings per analysis window
SAMPLE_PERIOD_S = 0.2     # one summarised reading every 200 ms (matches firmware)

STATES = ["HEALTHY", "WARNING", "CRITICAL"]
FAULTS = ["bearing_wear", "imbalance", "overheating", "overload", "voltage_fault"]

# Failure-risk bands applied to risk_score in [0, 1]
RISK_LOW_MAX = 0.25
RISK_MEDIUM_MAX = 0.60
