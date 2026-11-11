"""Synthetic motor-sensor data generator.

Used for software-first development before real hardware is connected.
Each window simulates WINDOW_SIZE readings of six sensors.  A continuous
`severity` value controls how far a fault has progressed:
    0.0 = healthy, ~0.25-0.75 = warning, >= 0.8 = critical.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FAULTS, SAMPLE_PERIOD_S, SENSORS, STATES, WINDOW_SIZE
from .features import extract_features
from .preprocessing import clean_window

BASELINE = {"vibration": 2.0, "temperature": 55.0, "current": 10.0,
            "voltage": 415.0, "speed": 1450.0, "pressure": 4.0}
NOISE = {"vibration": 0.15, "temperature": 0.4, "current": 0.25,
         "voltage": 2.0, "speed": 5.0, "pressure": 0.05}

WARNING_SEVERITY = 0.2   # below this the motor is considered healthy
CRITICAL_SEVERITY = 0.8


def severity_to_state(severity: float) -> str:
    if severity < WARNING_SEVERITY:
        return "HEALTHY"
    return "WARNING" if severity < CRITICAL_SEVERITY else "CRITICAL"


def sample_severity(state: str, rng: np.random.Generator) -> float:
    if state == "HEALTHY":
        return 0.0
    if state == "WARNING":
        return float(rng.uniform(0.25, 0.75))
    return float(rng.uniform(0.85, 1.4))


def generate_window(severity: float = 0.0, fault: str = "bearing_wear",
                    rng: np.random.Generator | None = None, window: int = WINDOW_SIZE,
                    start_ms: int = 0, dirty: bool = True) -> pd.DataFrame:
    """Return a raw (uncleaned) window as a DataFrame [timestamp + sensors]."""
    rng = rng or np.random.default_rng()
    n = window
    t = np.arange(n) * SAMPLE_PERIOD_S

    # Normal operating-point variation between windows
    base = dict(BASELINE)
    base["vibration"] *= rng.normal(1.0, 0.05)
    base["temperature"] += rng.normal(0, 2.0)
    base["current"] *= rng.normal(1.0, 0.04)
    base["voltage"] += rng.normal(0, 3.0)
    base["speed"] += rng.normal(0, 8.0)
    base["pressure"] += rng.normal(0, 0.1)

    d = {k: base[k] + rng.normal(0, NOISE[k], n) for k in SENSORS}
    s = max(severity, 0.0)

    if s > 0:
        if fault == "bearing_wear":
            d["vibration"] += base["vibration"] * 2.2 * s
            spikes = rng.random(n) < min(0.08 * s, 0.3)
            d["vibration"] += spikes * rng.uniform(1.0, 3.0, n) * base["vibration"] * s
            d["temperature"] += 14 * s
        elif fault == "imbalance":
            d["vibration"] += base["vibration"] * 2.8 * s * (1 + 0.15 * np.sin(2 * np.pi * t / 1.5))
            d["speed"] += rng.normal(0, 8 * s, n)
        elif fault == "overheating":
            d["temperature"] += 40 * s + np.linspace(0, 6 * s, n)
            d["current"] += 1.5 * s
            d["vibration"] += 0.3 * s
        elif fault == "overload":
            d["current"] += 7 * s
            d["temperature"] += 18 * s
            d["speed"] -= 70 * s
            d["vibration"] += 0.5 * s
        elif fault == "voltage_fault":
            d["voltage"] -= 35 * s + rng.normal(0, 6 * s, n)
            d["current"] += 2.5 * s
            d["temperature"] += 6 * s
        else:
            raise ValueError(f"Unknown fault: {fault}")

    df = pd.DataFrame(d)
    df.insert(0, "timestamp", start_ms + (t * 1000).astype(int))

    if dirty:  # imperfect real-world data: dropouts and isolated noise spikes
        for col in SENSORS:
            miss = rng.random(n) < 0.01
            df.loc[miss, col] = np.nan
            spike = rng.random(n) < 0.004
            df.loc[spike, col] = df[col].mean() * rng.uniform(1.3, 1.8)
    return df


def generate_dataset(samples_per_class: int = 500, seed: int = 42) -> pd.DataFrame:
    """Feature table with labels: state, fault, severity."""
    rng = np.random.default_rng(seed)
    rows = []
    for state in STATES:
        for _ in range(samples_per_class):
            sev = sample_severity(state, rng)
            fault = "none" if state == "HEALTHY" else str(rng.choice(FAULTS))
            raw = generate_window(sev, fault if fault != "none" else "bearing_wear", rng)
            feats = extract_features(clean_window(raw))
            feats.update(state=state, fault=fault, severity=sev)
            rows.append(feats)
    return pd.DataFrame(rows)
