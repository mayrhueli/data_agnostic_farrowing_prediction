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

"""Pool per-sow instance tables (across every sow you have for a given
signal type) into one table, for cross-sow training — grouped by sow,
as opposed to the single-sow checks in case_study.py.
"""
import pandas as pd

from video_activity.ingestion import load_accelerometer_series, load_video_series, load_meta
from farrowing_prediction.training.instance_tables import build_instance_table

LOADERS = {"accelerometer": load_accelerometer_series, "video": load_video_series}


def build_pooled_table(signal: str, sows: list):
    """Build one instance table for ``signal`` ("accelerometer" or
    "video") across ``sows``, tagged with a ``sow`` column for grouped
    CV. Returns (table, errors); a sow that can't be processed (missing
    file, too little pre-farrowing history, ...) is skipped and recorded
    in ``errors`` rather than aborting the whole pool, mirroring
    suite.py's per-sow error isolation.
    """
    if signal not in LOADERS:
        raise ValueError(f"signal must be one of {list(LOADERS)}, got {signal!r}")
    loader = LOADERS[signal]

    tables = []
    errors = {}
    for sow in sows:
        try:
            series = loader(sow)
            meta = load_meta(sow)
            close_crate = meta["close_crate"] if pd.notna(meta["close_crate"]) else None
            open_crate = meta["open_crate"] if pd.notna(meta["open_crate"]) else None
            table = build_instance_table(series, meta["start_farrowing"], close_crate, open_crate).copy()
            table["sow"] = sow
            tables.append(table)
        except Exception as e:  # noqa: BLE001 - one bad sow must not empty the whole pool
            errors[sow] = f"{type(e).__name__}: {e}"

    if not tables:
        return pd.DataFrame(), errors
    return pd.concat(tables, ignore_index=True), errors


def per_sow_baseline(raw_series: pd.DataFrame, entry_time: pd.Timestamp, window_hours: float = 48) -> float:
    """Mean raw activity over the first ``window_hours`` after pen entry —
    a sow's own resting/pre-nest-building baseline. Unlike domain-level
    scaling, this needs only that one animal's own early data, so it's
    available online for a brand-new sow rather than requiring the whole
    video population's statistics in advance.
    """
    window = raw_series[
        (raw_series["timestamp"] >= entry_time)
        & (raw_series["timestamp"] < entry_time + pd.Timedelta(hours=window_hours))
    ]
    return window["Acc"].mean()


def compute_per_sow_baselines(signal: str, sows: list, window_hours: float = 48) -> tuple:
    """``per_sow_baseline`` for every sow in ``sows``. Returns
    (baselines, errors); a sow with no data in the window (or any other
    load/parse failure) is skipped and recorded in ``errors`` rather than
    aborting the whole batch, mirroring ``build_pooled_table``.
    """
    if signal not in LOADERS:
        raise ValueError(f"signal must be one of {list(LOADERS)}, got {signal!r}")
    loader = LOADERS[signal]

    baselines = {}
    errors = {}
    for sow in sows:
        try:
            series = loader(sow)
            meta = load_meta(sow)
            baseline = per_sow_baseline(series, meta["entry_farrowing_pen"], window_hours=window_hours)
            if pd.isna(baseline):
                raise ValueError(f"no {signal} data in the {window_hours}h window after pen entry")
            baselines[sow] = baseline
        except Exception as e:  # noqa: BLE001 - one bad sow must not abort the batch
            errors[sow] = f"{type(e).__name__}: {e}"

    return baselines, errors
