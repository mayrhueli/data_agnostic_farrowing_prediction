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

"""Generates the fully synthetic "sample_sow" bundled under data/raw/ —
not real animal data, just a stand-in with the same file layout and
column names the real raw data has, so the pipeline (nest-building
detection, feature extraction, the full video_activity suite/figures)
is runnable out of the box without any real dataset.

Shape of the synthetic signal: low, noisy baseline activity; a ramp up
to a peak over the 36h before the synthetic farrowing time (standing in
for nest-building restlessness); a quiet period during "farrowing"
itself; then a settle to a slightly elevated post-farrowing baseline.
Accelerometer is written at the real 1-minute resolution; the video
series is written at a coarser-than-real 30s native resolution purely
to keep the bundled file small — ingestion.py resamples either to a
1-minute mean regardless of native rate, so this doesn't change how the
pipeline reads it.

Run from the repo root to (re)generate the bundled sample:

    python scripts/generate_sample_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SOW = "sample_sow"

ENTRY = pd.Timestamp("2024-01-01 00:00:00")
FARROWING = pd.Timestamp("2024-01-06 00:00:00")          # 5 days after entry
END_FARROWING = FARROWING + pd.Timedelta(hours=3)
CLOSE_CRATE = FARROWING - pd.Timedelta(hours=24)
OPEN_CRATE = FARROWING + pd.Timedelta(hours=8)
RECORDING_END = FARROWING + pd.Timedelta(hours=24)       # 1 day past farrowing


def _activity_curve(timestamps: pd.DatetimeIndex, peak: float, baseline: float, noise_std: float,
                     rng: np.random.Generator) -> np.ndarray:
    """Baseline -> ramp to `peak` over the 36h before FARROWING -> quiet
    during farrowing -> settle to a slightly elevated baseline after."""
    minutes_to_farrowing = (FARROWING - timestamps).total_seconds() / 60
    ramp_minutes = 36 * 60

    activity = np.full(len(timestamps), baseline, dtype=float)
    ramping = (minutes_to_farrowing > 0) & (minutes_to_farrowing <= ramp_minutes)
    progress = 1 - (minutes_to_farrowing[ramping] / ramp_minutes)
    activity[ramping] = baseline + (peak - baseline) * progress

    during_farrowing = (timestamps >= FARROWING) & (timestamps <= END_FARROWING)
    activity[during_farrowing] = baseline * 0.3

    after = timestamps > END_FARROWING
    activity[after] = baseline * 1.3

    noise = rng.normal(0, noise_std, size=len(timestamps))
    return np.clip(activity + noise, 0, None)


def generate_accelerometer(seed: int = 0) -> pd.DataFrame:
    timestamps = pd.date_range(ENTRY, RECORDING_END, freq="1min")
    rng = np.random.default_rng(seed)
    acc = _activity_curve(timestamps, peak=9.0, baseline=2.0, noise_std=0.3, rng=rng)
    return pd.DataFrame({"timestamp": timestamps, "Acc": acc.round(3)})


def generate_video(seed: int = 1) -> pd.DataFrame:
    timestamps = pd.date_range(ENTRY, RECORDING_END, freq="30s")
    rng = np.random.default_rng(seed)
    flow = _activity_curve(timestamps, peak=350_000, baseline=90_000, noise_std=8_000, rng=rng)
    return pd.DataFrame({"Timestamp": timestamps, "Flow_totals": flow.round(1)})


def generate_meta() -> pd.DataFrame:
    return pd.DataFrame([{
        "Sow.number.trial.number": SOW,
        "Date.of.sow.introduction.to.farrowing.pen": ENTRY.date().isoformat(),
        "Datetime.of.sows.farrowing": str(FARROWING),
        "end.of.farrowing": str(END_FARROWING),
        "pen.type": "F",
        "closing.crate": str(CLOSE_CRATE),
        "opening.crate": str(OPEN_CRATE),
        "Smartbow.tag.number": 999,
    }])


def main():
    acc_dir = REPO_ROOT / "data" / "raw" / "accelerometer"
    video_dir = REPO_ROOT / "data" / "raw" / "video"
    meta_dir = REPO_ROOT / "data" / "raw" / "meta"
    for d in (acc_dir, video_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    generate_accelerometer().to_csv(acc_dir / f"{SOW}.csv", index=False)
    generate_video().to_csv(video_dir / f"{SOW}_activity.csv", index=False)
    generate_meta().to_csv(meta_dir / f"{SOW}.csv", index=False)

    print(f"Wrote synthetic sample sow '{SOW}' to {acc_dir}, {video_dir}, {meta_dir}")


if __name__ == "__main__":
    main()
