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

"""Loaders that turn raw sensor exports into the standardised 1-minute
``timestamp, Acc`` series that the rest of the pipeline (nest-building
detection, feature engineering) already expects.

Two source signal types are supported for a given sow:

- accelerometer: ear-tag data, delivered as
  ``data/raw/accelerometer/<sow>.csv`` with columns ``timestamp,Acc`` at
  1-minute resolution.
- video: computer-vision-derived activity, delivered as
  ``data/raw/video/<sow>_activity.csv`` with columns ``Timestamp,Flow_totals``
  at approximately 1 Hz. It is resampled to a 1-minute mean so it can be fed
  into the exact same downstream code (CUSUM/BEAST/NestDetect, feature
  extraction) as the accelerometer signal.
"""
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ACCELEROMETER_DIR = REPO_ROOT / "data" / "raw" / "accelerometer"
VIDEO_DIR = REPO_ROOT / "data" / "raw" / "video"
META_DIR = REPO_ROOT / "data" / "raw" / "meta"


def load_accelerometer_series(sow: str, base_dir: Path = ACCELEROMETER_DIR) -> pd.DataFrame:
    """Load the 1-minute accelerometer magnitude series for a sow."""
    path = base_dir / f"{sow}.csv"
    df = pd.read_csv(path)
    df.columns = [c.strip().strip('"') for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df[["timestamp", "Acc"]]
    df["source"] = "accelerometer"
    return df


def load_video_series(sow: str, base_dir: Path = VIDEO_DIR, resample_rule: str = "1min") -> pd.DataFrame:
    """Load the raw ~1 Hz video-derived optical-flow activity series for a
    sow and resample it to a 1-minute mean, renaming the ``Flow_totals``
    column to ``Acc`` so it is a drop-in replacement for the accelerometer
    magnitude in every downstream step (rolling averages, CUSUM, BEAST,
    NestDetect, feature extraction).
    """
    path = base_dir / f"{sow}_activity.csv"
    df = pd.read_csv(path)
    df.columns = [c.strip().strip('"') for c in df.columns]
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="mixed")
    df = df.sort_values("Timestamp").reset_index(drop=True)

    resampled = (
        df.set_index("Timestamp")["Flow_totals"]
        .resample(resample_rule)
        .mean()
        .rename("Acc")
        .to_frame()
        .reset_index()
        .rename(columns={"Timestamp": "timestamp"})
    )
    resampled["source"] = "video"
    return resampled


def load_meta(sow: str, base_dir: Path = META_DIR) -> pd.Series:
    """Load the per-sow metadata row (farrowing onset, crate times, pen
    type, ...) and parse the datetime columns.
    """
    path = base_dir / f"{sow}.csv"
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    row = df.iloc[0]

    parsed = {
        "sow": str(row["Sow.number.trial.number"]).strip(),
        "entry_farrowing_pen": pd.to_datetime(row["Date.of.sow.introduction.to.farrowing.pen"]),
        "start_farrowing": pd.to_datetime(row["Datetime.of.sows.farrowing"]),
        "end_farrowing": pd.to_datetime(row["end.of.farrowing"]),
        "pen_type": row["pen.type"],
        # closing.crate/opening.crate hold a real, non-missing sentinel value
        # ("uncrated") for sows that were never crated, rather than being
        # blank/NaN — errors="coerce" turns that (or any other unparseable
        # value) into NaT, which build_instance_table already treats as
        # "not crated", the correct reading of "uncrated" anyway.
        "close_crate": pd.to_datetime(row["closing.crate"], errors="coerce"),
        "open_crate": pd.to_datetime(row["opening.crate"], errors="coerce"),
        "smartbow_tag": row.get("Smartbow.tag.number"),
    }
    return pd.Series(parsed)
