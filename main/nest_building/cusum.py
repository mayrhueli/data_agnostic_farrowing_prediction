# Copyright (C) 2026 Elisabeth Mayrhuber

"""CUSUM changepoint detection for nest-building onset with parameters
k=2 and threshold=150. Runs on the 12h rolling average of the activity
signal, starting two days after the sow enters the farrowing pen.
"""
import numpy as np
import pandas as pd

K = 2.0
THRESHOLD = 150.0
HALF_DAY = 720  # 12h in minutes, matches the 12h rolling average window
START_OFFSET = 2 * 1440  # detection starts 2 days after pen entry


def detect_cusum(df: pd.DataFrame, k: float = K, threshold: float = THRESHOLD,
                  start_offset: int = START_OFFSET):
    """Run CUSUM on a 1-minute ``timestamp, Acc`` series.

    Returns (changepoint_timestamp, s_plus, s_minus) where the two arrays
    are the upward/downward cumulative sums for the whole rolling-mean
    series (useful for plotting), and changepoint_timestamp is None if no
    alarm ever fired.
    """
    rolling = df["Acc"].rolling(window=HALF_DAY, min_periods=1).mean()

    s_plus = np.zeros(len(df))
    s_minus = np.zeros(len(df))
    changepoint_idx = None

    running_mean = rolling.expanding().mean()
    running_std = rolling.expanding().std().replace(0, np.nan)

    for t in range(1, len(df)):
        if t < start_offset:
            continue
        mu = running_mean.iloc[t - 1]
        sigma = running_std.iloc[t - 1]
        if pd.isna(sigma) or sigma == 0:
            continue
        deviation = (rolling.iloc[t] - mu) / sigma
        s_plus[t] = max(0.0, s_plus[t - 1] + deviation - k)
        s_minus[t] = max(0.0, s_minus[t - 1] - deviation - k)

        if changepoint_idx is None and (s_plus[t] > threshold or s_minus[t] > threshold):
            changepoint_idx = t
            break

    changepoint_ts = df["timestamp"].iloc[changepoint_idx] if changepoint_idx is not None else None
    return changepoint_ts, s_plus, s_minus, rolling
