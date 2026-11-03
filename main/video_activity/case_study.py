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

"""Case-study pipeline: run nest-building detection (CUSUM, BEAST,
NestDetect) on both the accelerometer signal and the video-derived
(optical-flow) activity signal for one sow, and compare them.

Usage (from the repo root):

    python -m video_activity.case_study --sow "sample_sow"

Designed to run per-sow so it can be looped over once more paired
accelerometer/video sows are added under data/raw/{accelerometer,video}/
and data/raw/meta/.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from video_activity.ingestion import load_accelerometer_series, load_video_series, load_meta
from nest_building.nestdetect import detect_nest_building as run_nestdetect
from nest_building.cusum import detect_cusum
from nest_building.beast import detect_beast, RBEAST_AVAILABLE
from nest_building.evaluation import evaluate_detection

OUT_DIR = Path(__file__).resolve().parents[1] / "out" / "video_case_study"


def crop_series(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
    return df.loc[mask].reset_index(drop=True)


def run_detection_methods(df: pd.DataFrame, signal_name: str, farrowing_time: pd.Timestamp):
    """Run all three detectors, isolating failures per method: a pathological
    signal that breaks e.g. BEAST shouldn't also cost the CUSUM/NestDetect
    results for the same sow.
    """
    results = {}

    def _run(method, fn):
        try:
            cp = fn()
        except Exception as e: 
            print(f"  {method} failed on {signal_name}: {type(e).__name__}: {e}")
            results[method] = None
            return
        results[method] = evaluate_detection(method, signal_name, cp, farrowing_time)

    _run("NestDetect", lambda: run_nestdetect(df)[0])
    _run("CUSUM", lambda: detect_cusum(df)[0])
    if RBEAST_AVAILABLE:
        _run("BEAST", lambda: detect_beast(df)[0])
    else:
        results["BEAST"] = None

    return results


def plot_detection(acc_df, video_df, detections, farrowing_time, sow, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    colors = {"NestDetect": "tab:red", "CUSUM": "tab:orange", "BEAST": "tab:purple"}

    for ax, df, signal_name, display_name in zip(
        axes, [acc_df, video_df], ["accelerometer", "video"], ["accelerometer", "video (optical flow)"]
    ):
        rolling_12h = df["Acc"].rolling(window=720, min_periods=1).mean()
        days_to_farrowing = (df["timestamp"] - farrowing_time).dt.total_seconds() / 86400
        ax.plot(days_to_farrowing, rolling_12h, color="tab:blue", label="rolling mean 12h", linewidth=1)

        end_farrowing = None
        for method, det in detections[signal_name].items():
            if det is None or det.changepoint is None:
                continue
            x = (det.changepoint - farrowing_time).total_seconds() / 86400
            ax.axvline(x, color=colors[method], linestyle="--", label=f"{method} changepoint")

        ax.axvspan(-1, 0, color="grey", alpha=0.15, label="24h before farrowing")
        ax.axvline(0, color="black", linewidth=1.5, label="farrowing onset")
        ax.set_title(f"Sow {sow} — {display_name}")
        ax.set_ylabel("Rolling activity [12h mean]")
        ax.legend(loc="upper left", fontsize=8)

    axes[-1].set_xlabel("Days relative to onset of farrowing")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(sow: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = load_meta(sow)
    farrowing_time = meta["start_farrowing"]
    entry = meta["entry_farrowing_pen"]

    acc_raw = load_accelerometer_series(sow)
    video_raw = load_video_series(sow)

    detection_end = farrowing_time + pd.Timedelta(hours=12)
    acc_det = crop_series(acc_raw, entry, detection_end)
    video_det = crop_series(video_raw, entry, detection_end)

    detections = {
        "accelerometer": run_detection_methods(acc_det, "accelerometer", farrowing_time),
        "video": run_detection_methods(video_det, "video", farrowing_time),
    }

    plot_detection(acc_det, video_det, detections, farrowing_time, sow, OUT_DIR / f"{sow}_detection.png")

    summary = {
        "sow": sow,
        "farrowing_time": str(farrowing_time),
        "pen_type": meta["pen_type"],
        "detection": {
            signal: {
                method: (
                    None if det is None else {
                        "changepoint": str(det.changepoint) if det.changepoint is not None else None,
                        "lead_time_hours": det.lead_time_hours,
                        "classification_24h": det.classification_24h,
                        "classification_48h": det.classification_48h,
                    }
                )
                for method, det in methods.items()
            }
            for signal, methods in detections.items()
        },
    }

    with open(OUT_DIR / f"{sow}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the case study for one sow. "
                                                  "For every paired sow you have, use video_activity.suite instead.")
    parser.add_argument("--sow", default="sample_sow", help="Single sow id, e.g. 'sample_sow'")
    args = parser.parse_args()
    main(args.sow)
