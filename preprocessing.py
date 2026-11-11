"""Data preprocessing: synchronisation, missing values, noise removal.

Normalisation / scaling is done inside the sklearn Pipeline (StandardScaler)
so that exactly the same scaling is applied at training and inference time.
"""
from __future__ import annotations

import pandas as pd

from .config import SENSORS


def synchronize_streams(streams: dict[str, pd.DataFrame], tolerance_ms: int = 100) -> pd.DataFrame:
    """Align independent sensor streams onto one timeline.

    streams: {sensor_name: DataFrame[timestamp(ms), value]}
    Uses the first stream as the master clock and attaches the nearest
    reading (within tolerance) from every other stream.
    """
    names = list(streams)
    out = streams[names[0]].rename(columns={"value": names[0]}).sort_values("timestamp")
    for name in names[1:]:
        right = streams[name].rename(columns={"value": name}).sort_values("timestamp")
        out = pd.merge_asof(out, right, on="timestamp", direction="nearest", tolerance=tolerance_ms)
    return out.reset_index(drop=True)


def clean_window(df: pd.DataFrame, sensors: list[str] = SENSORS) -> pd.DataFrame:
    """Clean one analysis window.

    1. Sort by timestamp (sensor synchronisation)
    2. Coerce to numeric and fill missing values by interpolation
    3. 3-point rolling median filter (removes isolated noise spikes)
    """
    df = df.copy()
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
    cols = [c for c in sensors if c in df.columns]
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    df[cols] = df[cols].interpolate(limit_direction="both")
    df[cols] = df[cols].rolling(3, center=True, min_periods=1).median()
    return df
