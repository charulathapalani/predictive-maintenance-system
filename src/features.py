"""Feature extraction from one cleaned sensor window."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SAMPLE_PERIOD_S, SENSORS

FEATURE_NAMES = ["mean", "rms", "std", "var", "peak", "p2p", "max", "min", "roc"]
FEATURE_COLUMNS = [f"{s}_{f}" for s in SENSORS for f in FEATURE_NAMES]


def signal_features(x) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    diffs = np.diff(x)
    return {
        "mean": float(np.mean(x)),
        "rms": float(np.sqrt(np.mean(x ** 2))),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "var": float(np.var(x, ddof=1)) if len(x) > 1 else 0.0,
        "peak": float(np.max(np.abs(x))),
        "p2p": float(np.ptp(x)),
        "max": float(np.max(x)),
        "min": float(np.min(x)),
        "roc": float(np.mean(np.abs(diffs)) / SAMPLE_PERIOD_S) if len(diffs) else 0.0,
    }


def extract_features(df: pd.DataFrame, sensors: list[str] = SENSORS) -> dict[str, float]:
    """Return {'<sensor>_<feature>': value}. Missing sensors give NaN."""
    feats: dict[str, float] = {}
    for s in sensors:
        if s in df.columns and df[s].notna().any():
            for name, val in signal_features(df[s].dropna().values).items():
                feats[f"{s}_{name}"] = val
        else:
            for name in FEATURE_NAMES:
                feats[f"{s}_{name}"] = np.nan
    return feats
