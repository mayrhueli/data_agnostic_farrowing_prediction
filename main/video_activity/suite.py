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

"""Batch suite: run the accelerometer-vs-video nest-building detection
case study over every paired sow you have locally, and aggregate the
results.

Usage:

    python -m video_activity.suite                 # run every complete pair
    python -m video_activity.suite --sow "ID" --sow "ID"

Drop new sows in as ``data/raw/accelerometer/<sow>.csv``,
``data/raw/video/<sow>_activity.csv`` and ``data/raw/meta/<sow>.csv`` (see
video_activity/README.md) and re-run; a sow missing any of the three
files is reported and skipped rather than failing the whole run, and a
sow that raises during processing (bad timestamps, too little data
before farrowing, ...) is caught and reported the same way so one bad
pair doesn't stop the rest.
"""
import argparse
import json
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from video_activity import case_study
from video_activity.ingestion import ACCELEROMETER_DIR, VIDEO_DIR, META_DIR

SUITE_DIR = case_study.OUT_DIR / "suite"

DETECTION_METHODS = ["NestDetect", "CUSUM", "BEAST"]


def discover_pairs() -> dict:
    """Every sow id referenced by any of the three data folders, with
    which of {meta, accelerometer, video} are actually present for it.
    """
    meta_sows = {p.stem for p in META_DIR.glob("*.csv")}
    acc_sows = {p.stem for p in ACCELEROMETER_DIR.glob("*.csv")}
    video_sows = {p.name[: -len("_activity.csv")] for p in VIDEO_DIR.glob("*_activity.csv")}

    all_sows = meta_sows | acc_sows | video_sows
    return {
        sow: {"meta": sow in meta_sows, "accelerometer": sow in acc_sows, "video": sow in video_sows}
        for sow in sorted(all_sows)
    }


def complete_pairs(pairs: dict = None) -> list:
    """Sow ids that have all three files and can actually be run."""
    pairs = pairs if pairs is not None else discover_pairs()
    return sorted(sow for sow, present in pairs.items() if all(present.values()))


def incomplete_pairs(pairs: dict = None) -> dict:
    """{sow: [missing file types]} for sows that can't be run yet."""
    pairs = pairs if pairs is not None else discover_pairs()
    return {
        sow: [kind for kind, present in flags.items() if not present]
        for sow, flags in pairs.items()
        if not all(flags.values())
    }


def detection_long(results: dict) -> pd.DataFrame:
    """Tidy one row per (sow, signal, method) detection outcome."""
    rows = []
    for sow, summary in results.items():
        for signal, methods in summary.get("detection", {}).items():
            for method, det in methods.items():
                if det is None:
                    continue
                rows.append({
                    "sow": sow,
                    "signal": signal,
                    "method": method,
                    "classification_24h": det["classification_24h"],
                    "classification_48h": det["classification_48h"],
                    "lead_time_hours": det["lead_time_hours"],
                })
    return pd.DataFrame(rows, columns=["sow", "signal", "method", "classification_24h",
                                        "classification_48h", "lead_time_hours"])


def detection_stats(long_df: pd.DataFrame) -> pd.DataFrame:
    """Per (method, signal): % early/on_time/late at 24h and 48h, and mean
    lead time among on-time detections.
    """
    if long_df.empty:
        return pd.DataFrame(columns=["method", "signal", "n", "early_24h_pct", "on_time_24h_pct",
                                      "late_24h_pct", "early_48h_pct", "on_time_48h_pct", "late_48h_pct",
                                      "mean_lead_time_on_time_48h_h"])

    rows = []
    for (method, signal), group in long_df.groupby(["method", "signal"]):
        n = len(group)
        row = {"method": method, "signal": signal, "n": n}
        for window in ["24h", "48h"]:
            counts = group[f"classification_{window}"].value_counts(normalize=True) * 100
            for cls in ["early", "on_time", "late"]:
                row[f"{cls}_{window}_pct"] = round(counts.get(cls, 0.0), 1)
        on_time_48h = group[group["classification_48h"] == "on_time"]
        row["mean_lead_time_on_time_48h_h"] = round(on_time_48h["lead_time_hours"].mean(), 2) if len(on_time_48h) else None
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["method", "signal"]).reset_index(drop=True)


def plot_detection_summary(det_stats: pd.DataFrame, out_path: Path):
    if det_stats.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    signals = sorted(det_stats["signal"].unique())
    x = np.arange(len(DETECTION_METHODS))
    width = 0.8 / max(len(signals), 1)

    for i, signal in enumerate(signals):
        sub = det_stats[det_stats["signal"] == signal].set_index("method").reindex(DETECTION_METHODS)
        ax.bar(x + i * width, sub["on_time_48h_pct"].fillna(0), width, label=signal)

    ax.set_xticks(x + width * (len(signals) - 1) / 2)
    ax.set_xticklabels(DETECTION_METHODS)
    ax.set_ylabel("Detections classified \"on time\" (48h window) [%]")
    ax.set_title(f"Nest-building detection across {det_stats['n'].iloc[0] if len(det_stats) else 0} sow-method-signal cells")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _df_to_markdown_or_text(df: pd.DataFrame) -> str:
    """df.to_markdown() needs the optional `tabulate` package at a
    version pandas is happy with (>=0.9.0); a stale install raises
    ImportError deep inside pandas. That's just a reporting nicety, not
    worth losing the whole suite run over after the actual analysis
    already succeeded, so fall back to a plain fixed-width table in a
    code fence — still readable in the .md file, just not a real
    Markdown table.
    """
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```\n" + df.to_string(index=False) + "\n```"


def write_report(sows_run, det_stats, errors, missing, out_path: Path):
    lines = ["# Video-vs-accelerometer suite report", ""]

    lines.append(f"Sows run successfully: **{len(sows_run)}**")
    lines.append("")

    if missing:
        lines.append("## Skipped (incomplete data)")
        for sow, kinds in sorted(missing.items()):
            lines.append(f"- `{sow}`: missing {', '.join(kinds)}")
        lines.append("")

    if errors:
        lines.append("## Failed while processing")
        for sow, err in sorted(errors.items()):
            lines.append(f"- `{sow}`: {err}")
        lines.append("")

    lines.append("## Nest-building detection (% \"on time\" per method/signal)")
    lines.append("")
    lines.append(_df_to_markdown_or_text(det_stats) if not det_stats.empty else "_no results_")
    lines.append("")

    out_path.write_text("\n".join(lines))


def run_suite(sows: list = None) -> dict:
    pairs = discover_pairs()
    missing = incomplete_pairs(pairs)
    for sow, kinds in missing.items():
        print(f"Skipping {sow}: missing {', '.join(kinds)}")

    run_sows = sows if sows else complete_pairs(pairs)
    if not run_sows:
        print("No complete accelerometer+video+meta triplets found under data/raw. "
              "See video_activity/README.md for the expected layout.")
        return {"sows": [], "errors": {}, "missing": missing}

    results = {}
    errors = {}
    for sow in run_sows:
        print(f"\n=== Running {sow} ===")
        try:
            results[sow] = case_study.main(sow)
        except Exception as e:  # noqa: BLE001 - one bad pair must not stop the suite
            print(f"ERROR in {sow}: {type(e).__name__}: {e}")
            traceback.print_exc()
            errors[sow] = f"{type(e).__name__}: {e}"

    SUITE_DIR.mkdir(parents=True, exist_ok=True)

    det_long = detection_long(results)
    det_stats = detection_stats(det_long)

    det_long.to_csv(SUITE_DIR / "detection_long.csv", index=False)
    det_stats.to_csv(SUITE_DIR / "detection_summary.csv", index=False)

    plot_detection_summary(det_stats, SUITE_DIR / "detection_summary.png")

    write_report(list(results.keys()), det_stats, errors, missing, SUITE_DIR / "suite_report.md")

    with open(SUITE_DIR / "suite_results.json", "w") as f:
        json.dump({"sows": list(results.keys()), "errors": errors, "missing": missing},
                   f, indent=2, default=str)

    print(f"\nRan {len(results)}/{len(run_sows)} sows successfully. Skipped {len(missing)} incomplete, "
          f"{len(errors)} failed outright. Report written to {SUITE_DIR}/suite_report.md")

    return {
        "sows": list(results.keys()),
        "results": results,
        "errors": errors,
        "missing": missing,
        "detection_long": det_long,
        "detection_stats": det_stats,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sow", action="append", default=None,
                         help="Restrict to specific sow id(s); repeatable. Default: every complete pair found.")
    args = parser.parse_args()
    run_suite(args.sow)
