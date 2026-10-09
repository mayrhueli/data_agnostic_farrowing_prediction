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

"""Builds per-sow instance tables (ten features + a time-to-farrowing
target) from a 1-minute ``timestamp, Acc`` series, in two windowing
modes:

- ``build_instance_table``: a fixed window counting back from a known
  farrowing time — useful for evaluating the prediction stage on its
  own, independent of any detection step.
- ``build_online_instance_table``: a window starting from a given alarm
  time (e.g. a nest-building detector's changepoint) and running forward
  to farrowing — the deployment-realistic setting, where the model only
  ever sees data up to the current moment.

Both functions extract features via
``feature_engineering.time_features.calculate_features_for_timeseries``,
called on ``df.iloc[:i]`` at every step so no instance ever looks ahead
of its own timestamp; they differ only in how the window's start/end
indices are chosen.
"""
import pandas as pd

from feature_engineering.time_features import calculate_features_for_timeseries
from feature_engineering.constants import DAY, TWO_DAYS

FEATURE_COLUMNS = [
    "mean_day", "mean_night", "mean_24", "mean_48",
    "mean_last_12", "mean_last_24", "trend_last_12", "trend_last_24",
    "currently_crated", "crated_since",
]


def _build_instance_table_from_index_range(df: pd.DataFrame, farrowing_time: pd.Timestamp,
                                            idx_start: int, idx_end: int,
                                            close_crate: pd.Timestamp = None, open_crate: pd.Timestamp = None,
                                            step_minutes: int = 60, empty_error: str = None) -> pd.DataFrame:
    """Shared core of ``build_instance_table``/``build_online_instance_table``:
    slide a ``step_minutes`` window over row indices ``[idx_start, idx_end)``,
    extracting the activity features from ``df.iloc[:i]`` at each step — only
    data available up to that point, never a look-ahead — plus the crating
    features and the remaining-time target.
    """
    rows = []
    for i in range(idx_start, idx_end, step_minutes):
        window = df.iloc[:i]
        if len(window) < TWO_DAYS:
            continue
        feats = calculate_features_for_timeseries(window, farrowing_time, full_features=False)

        r = df.iloc[i]
        if close_crate is not None and open_crate is not None and pd.notna(close_crate) and close_crate < r["timestamp"] < open_crate:
            feats["currently_crated"] = True
            feats["crated_since"] = pd.Timedelta(r["timestamp"] - close_crate).total_seconds() / 60
        else:
            feats["currently_crated"] = False
            feats["crated_since"] = 0.0

        rows.append(feats)

    if not rows:
        raise ValueError(empty_error or (
            f"No instance had at least {TWO_DAYS} minutes of history yet "
            f"(recording starts {df['timestamp'].min() if len(df) else 'n/a'}, farrowing is {farrowing_time})."
        ))

    table = pd.concat(rows, ignore_index=True)
    table["currently_crated"] = table["currently_crated"].astype(int)

    # mean_day/mean_night are computed over a fixed calendar-day window
    # near the start of the recording; for a short or gappy series that
    # window can be entirely empty, producing NaN for every row rather
    # than just a few. Forward/back-fill instead of dropping rows for it
    # — dropping would silently empty the whole table instead of just
    # losing two weak features. A column that is NaN for literally every
    # row (ffill/bfill has nothing to propagate) falls back to 0 rather
    # than being left NaN, which would otherwise crash a scaler fit on
    # this table downstream.
    table[FEATURE_COLUMNS] = table[FEATURE_COLUMNS].ffill().bfill().fillna(0.0)

    table = table.dropna(subset=["remaining_minutes"]).reset_index(drop=True)
    if table.empty:
        raise ValueError("All extracted instances have an undefined remaining_minutes target.")
    return table


def build_instance_table(df: pd.DataFrame, farrowing_time: pd.Timestamp,
                          close_crate: pd.Timestamp = None, open_crate: pd.Timestamp = None,
                          days_before: int = 2, step_minutes: int = 60) -> pd.DataFrame:
    """Slide an hourly window over the last ``days_before`` days before
    farrowing and extract the ten features + remaining time, for a
    single already-loaded, single-signal series.

    This window is **farrowing-anchored**: it always starts exactly
    ``days_before`` days before the (here, already-known) farrowing time,
    regardless of whether or when a detector actually raised an alarm —
    useful for isolating the prediction stage on its own, but not a
    deployment-realistic test (a live system doesn't know the farrowing
    time in advance to window around). See ``build_online_instance_table``
    for that setting.
    """
    idx = df[df["timestamp"] >= farrowing_time].index.values
    if len(idx) == 0:
        raise ValueError(
            f"Series does not reach the farrowing timestamp ({farrowing_time}): "
            f"last available sample is {df['timestamp'].max() if len(df) else 'n/a'}"
        )
    idx_before = max(0, idx[0] - days_before * DAY)
    return _build_instance_table_from_index_range(
        df, farrowing_time, idx_before, idx[0], close_crate, open_crate, step_minutes,
        empty_error=(
            f"No {days_before * 24}h-before-farrowing window has at least {TWO_DAYS} "
            f"minutes of history (recording starts {df['timestamp'].min() if len(df) else 'n/a'}, "
            f"farrowing is {farrowing_time}) — not enough pre-farrowing data to build any instance."
        ),
    )


def build_online_instance_table(df: pd.DataFrame, start_time: pd.Timestamp, farrowing_time: pd.Timestamp,
                                 close_crate: pd.Timestamp = None, open_crate: pd.Timestamp = None,
                                 step_minutes: int = 60) -> pd.DataFrame:
    """The **online** setting: slide an hourly window from ``start_time``
    (a detector's alarm time — the actual "go live" moment in a deployed
    system) up to ``farrowing_time``, rather than ``build_instance_table``'s
    fixed window counting back from the (in deployment, not yet known)
    farrowing time. Each instance still only uses data available up to
    that point (no look-ahead); only the *target* (``remaining_minutes``)
    and the *window's end* use the true farrowing time, which is standard
    for scoring a prediction against ground truth and is not information
    the model's features see.

    Raises if ``start_time`` is at or after ``farrowing_time`` — a
    detector that fires too late (or never) has no online prediction
    window at all, and that is itself the result worth reporting, not
    something to silently paper over.
    """
    if start_time >= farrowing_time:
        raise ValueError(
            f"start_time ({start_time}) is at or after farrowing_time ({farrowing_time}) — "
            f"no online prediction window exists (the alarm fired too late, or never fired)."
        )

    farrow_idx = df[df["timestamp"] >= farrowing_time].index.values
    if len(farrow_idx) == 0:
        raise ValueError(
            f"Series does not reach the farrowing timestamp ({farrowing_time}): "
            f"last available sample is {df['timestamp'].max() if len(df) else 'n/a'}"
        )
    start_idx = df[df["timestamp"] >= start_time].index.values
    if len(start_idx) == 0:
        raise ValueError(
            f"Series does not reach start_time ({start_time}): "
            f"last available sample is {df['timestamp'].max() if len(df) else 'n/a'}"
        )

    return _build_instance_table_from_index_range(
        df, farrowing_time, start_idx[0], farrow_idx[0], close_crate, open_crate, step_minutes,
        empty_error=(
            f"No instance in the [{start_time}, {farrowing_time}) online window has at least "
            f"{TWO_DAYS} minutes of history yet (recording starts "
            f"{df['timestamp'].min() if len(df) else 'n/a'}) — not enough history since the alarm "
            f"fired to build any online instance."
        ),
    )
