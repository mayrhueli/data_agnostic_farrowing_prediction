# Copyright (C) 2026 Elisabeth Mayrhuber
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""BEAST changepoint detection for nest-building onset: periodicity of
one day, periodic seasonality, at most one changepoint. Uses the
``Rbeast`` package published by the original BEAST authors (Zhao,
2019/2025). BEAST is batch (needs the whole signal) and nondeterministic
(MCMC), unlike CUSUM/NestDetect.
"""
import numpy as np
import pandas as pd

try:
    import Rbeast as rb
    RBEAST_AVAILABLE = True
except ImportError:  
    RBEAST_AVAILABLE = False

START_OFFSET = 2 * 1440  # detection starts 2 days after pen entry


def detect_beast(df: pd.DataFrame, start_offset: int = START_OFFSET, mcmc_seed: int = 0):
    """Run BEAST on a 1-minute ``timestamp, Acc`` series (from
    ``start_offset`` minutes onward) and return the changepoint timestamp
    with the highest occurrence probability, or None if BEAST is
    unavailable or finds none.
    """
    if not RBEAST_AVAILABLE:
        return None, None

    signal = df["Acc"].to_numpy()[start_offset:]
    timestamps = df["timestamp"].iloc[start_offset:].reset_index(drop=True)
    if len(signal) < 2 * 1440:
        return None, None

    out = rb.beast(
        signal,
        start=1,
        deltat=1.0 / 1440.0,
        season="harmonic",
        period=1.0,
        tcp_minmax=[0, 1],
        mcmc_seed=mcmc_seed,
        print_param=False,
        print_progress=False,
        print_warning=False,
        quiet=True,
    )

    cp = np.atleast_1d(out.trend.cp)
    cp_pr = np.atleast_1d(out.trend.cpPr)
    valid = ~np.isnan(cp)
    if not valid.any():
        return None, out

    cp = cp[valid]
    cp_pr = cp_pr[valid]
    best_idx = int(np.argmax(cp_pr))
    best_cp = cp[best_idx]

    idx = int(round(best_cp))
    idx = max(0, min(idx, len(timestamps) - 1))
    changepoint_ts = timestamps.iloc[idx]
    return changepoint_ts, out
