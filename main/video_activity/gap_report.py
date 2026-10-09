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

"""Recording-gap report: for each sow and signal (accelerometer/video),
find every gap in the 1-minute ``timestamp, Acc`` series and report how
many there are, their total duration, and where each one falls relative
to farrowing onset (before it / after it / spanning it, and how many
hours out). A good first read before trusting a detection or prediction
result — a gap sitting right before farrowing can explain a missed
changepoint or a noisy instance far more plausibly than a model failure.

A "gap" is a jump between consecutive *valid* readings bigger than
``tolerance`` x the expected 1-minute interval. Accelerometer gaps show
up as literally absent rows (``ingestion.load_accelerometer_series``
loads the file as-is); video gaps show up as ``NaN`` runs instead, since
``ingestion.load_video_series`` resamples onto a complete 1-minute grid
and fills missing minutes with ``NaN`` rather than dropping them. Both
are detected the same way here by dropping ``NaN`` Acc rows first, so
one gap-detection function covers both signal types without needing to
know which is which.

Usage:

    python -m video_activity.gap_report                       # every sow with accelerometer or video data
    python -m video_activity.gap_report --sow "147368 19" "X"  # restrict to specific sows

Output (``out/video_case_study/gap_report/``):

- ``gap_report_long.csv`` — one row per gap (sow, signal, gap_start,
  gap_end, duration_hours, hours_before_farrowing_start/_end, position).
- ``gap_report_summary.csv`` — one row per sow x signal (n_gaps,
  total_gap_hours, largest_gap_hours, mean_gap_hours).
- ``gap_report.md`` — a short human-readable summary.
"""
import argparse
from pathlib import Path

import pandas as pd

from video_activity import suite
from video_activity.ingestion import load_accelerometer_series, load_video_series, load_meta

OUT_DIR = suite.case_study.OUT_DIR / "gap_report"
LOADERS = {"accelerometer": load_accelerometer_series, "video": load_video_series}
EXPECTED_INTERVAL = pd.Timedelta(minutes=1)


def detect_gaps(series: pd.DataFrame, expected_interval: pd.Timedelta = EXPECTED_INTERVAL,
                 tolerance: float = 1.5) -> pd.DataFrame:
    """Every gap in ``series`` bigger than ``tolerance`` x
    ``expected_interval``. Returns a DataFrame with one row per gap:
    ``gap_start`` (last valid reading before it), ``gap_end`` (next
    valid reading after it), ``duration_hours`` (elapsed time between
    them — the amount of missing recording, not just the overshoot past
    one expected interval).
    """
    valid = series.dropna(subset=["Acc"]).sort_values("timestamp")
    if len(valid) < 2:
        return pd.DataFrame(columns=["gap_start", "gap_end", "duration_hours"])

    timestamps = valid["timestamp"].reset_index(drop=True)
    diffs = timestamps.diff()
    gap_idx = diffs.index[diffs > expected_interval * tolerance]
    if len(gap_idx) == 0:
        return pd.DataFrame(columns=["gap_start", "gap_end", "duration_hours"])

    gaps = pd.DataFrame({
        "gap_start": timestamps.loc[gap_idx - 1].to_numpy(),
        "gap_end": timestamps.loc[gap_idx].to_numpy(),
    })
    gaps["duration_hours"] = (gaps["gap_end"] - gaps["gap_start"]).dt.total_seconds() / 3600
    return gaps


def classify_position(gaps: pd.DataFrame, farrowing_time: pd.Timestamp) -> pd.DataFrame:
    """Add ``hours_before_farrowing_start``/``_end`` (positive = that
    many hours before farrowing onset; negative = after it) and
    ``position`` (``"before farrowing"``, ``"after farrowing"``, or
    ``"spans farrowing"``) columns."""
    gaps = gaps.copy()
    if gaps.empty:
        gaps["hours_before_farrowing_start"] = pd.Series(dtype=float)
        gaps["hours_before_farrowing_end"] = pd.Series(dtype=float)
        gaps["position"] = pd.Series(dtype=object)
        return gaps

    gaps["hours_before_farrowing_start"] = (farrowing_time - gaps["gap_start"]).dt.total_seconds() / 3600
    gaps["hours_before_farrowing_end"] = (farrowing_time - gaps["gap_end"]).dt.total_seconds() / 3600
    gaps["position"] = "spans farrowing"
    gaps.loc[gaps["gap_end"] <= farrowing_time, "position"] = "before farrowing"
    gaps.loc[gaps["gap_start"] >= farrowing_time, "position"] = "after farrowing"
    return gaps


def sows_with_any_signal(pairs: dict = None) -> list:
    """Every sow with an accelerometer or video file — metadata not
    required (gaps/totals are still reported without it; only the
    farrowing-relative columns need it)."""
    pairs = pairs if pairs is not None else suite.discover_pairs()
    return sorted(sow for sow, flags in pairs.items() if flags["accelerometer"] or flags["video"])


def gap_report_for_sow(sow: str, pairs_flags: dict = None) -> dict:
    """Gap report for one sow, isolating failures per signal (a broken
    video file must not cost the accelerometer result, and vice versa)
    and per sow (the caller does that — see ``run()``)."""
    flags = pairs_flags or {}
    try:
        meta = load_meta(sow)
        farrowing_time = meta["start_farrowing"]
    except Exception as e:  # noqa: BLE001 - missing/bad meta shouldn't block the per-signal gap counts
        farrowing_time = None
        meta_error = f"{type(e).__name__}: {e}"
    else:
        meta_error = None

    report = {"sow": sow, "farrowing_time": str(farrowing_time) if farrowing_time is not None else None,
              "meta_error": meta_error}

    for signal_name, loader in LOADERS.items():
        if flags and not flags.get(signal_name, True):
            continue  # file doesn't exist for this sow/signal — not an error, just not present
        try:
            series = loader(sow)
            gaps = detect_gaps(series)
            if farrowing_time is not None:
                gaps = classify_position(gaps, farrowing_time)
            report[signal_name] = {
                "n_gaps": len(gaps),
                "total_gap_hours": float(gaps["duration_hours"].sum()) if not gaps.empty else 0.0,
                "largest_gap_hours": float(gaps["duration_hours"].max()) if not gaps.empty else 0.0,
                "mean_gap_hours": float(gaps["duration_hours"].mean()) if not gaps.empty else 0.0,
                "gaps": gaps.to_dict("records"),
            }
        except Exception as e:  # noqa: BLE001 - one bad signal file must not cost the other
            report[signal_name] = {"error": f"{type(e).__name__}: {e}"}

    return report


def to_long_df(reports: list) -> pd.DataFrame:
    """One row per gap across every sow/signal in ``reports`` (the list
    ``run()``/``gap_report_for_sow`` produces)."""
    rows = []
    for report in reports:
        sow = report["sow"]
        for signal_name in LOADERS:
            res = report.get(signal_name)
            if not res or "error" in res or not res["gaps"]:
                continue
            for gap in res["gaps"]:
                rows.append({"sow": sow, "signal": signal_name, **gap})
    if not rows:
        return pd.DataFrame(columns=["sow", "signal", "gap_start", "gap_end", "duration_hours",
                                      "hours_before_farrowing_start", "hours_before_farrowing_end", "position"])
    return pd.DataFrame(rows)


def to_summary_df(reports: list) -> pd.DataFrame:
    """One row per sow x signal: n_gaps, total/largest/mean gap hours."""
    rows = []
    for report in reports:
        sow = report["sow"]
        for signal_name in LOADERS:
            res = report.get(signal_name)
            if not res:
                continue
            if "error" in res:
                rows.append({"sow": sow, "signal": signal_name, "error": res["error"]})
                continue
            rows.append({
                "sow": sow, "signal": signal_name, "error": None,
                "n_gaps": res["n_gaps"], "total_gap_hours": res["total_gap_hours"],
                "largest_gap_hours": res["largest_gap_hours"], "mean_gap_hours": res["mean_gap_hours"],
            })
    return pd.DataFrame(rows)


def write_report(reports: list, long_df: pd.DataFrame, summary_df: pd.DataFrame, out_dir: Path) -> str:
    lines = ["# Recording gap report", ""]
    lines.append(f"{len(reports)} sow(s) checked.")
    meta_missing = [r["sow"] for r in reports if r.get("meta_error")]
    if meta_missing:
        lines.append(f"\n{len(meta_missing)} sow(s) have no usable metadata, so their gaps are counted "
                      f"but not positioned relative to farrowing: {meta_missing}")

    if not summary_df.empty:
        clean = summary_df[summary_df["error"].isna()] if "error" in summary_df.columns else summary_df
        if not clean.empty:
            lines.append("\n## Summary (sow x signal)\n")
            try:
                lines.append(clean.to_markdown(index=False))
            except ImportError:
                lines.append("```\n" + clean.to_string(index=False) + "\n```")

        errored = summary_df[summary_df["error"].notna()] if "error" in summary_df.columns else summary_df.iloc[0:0]
        if not errored.empty:
            lines.append("\n## Failed to load\n")
            for _, row in errored.iterrows():
                lines.append(f"- {row['sow']} / {row['signal']}: {row['error']}")

    if not long_df.empty:
        lines.append(f"\n## Gaps closest to farrowing (within 48h, either side)\n")
        near = long_df[long_df["hours_before_farrowing_start"].abs().le(48)
                        | long_df["hours_before_farrowing_end"].abs().le(48)] if \
            "hours_before_farrowing_start" in long_df.columns else long_df.iloc[0:0]
        if not near.empty:
            cols = ["sow", "signal", "duration_hours", "hours_before_farrowing_start", "position"]
            try:
                lines.append(near[cols].sort_values("hours_before_farrowing_start").to_markdown(index=False))
            except ImportError:
                lines.append("```\n" + near[cols].to_string(index=False) + "\n```")
        else:
            lines.append("None.")

    text = "\n".join(lines) + "\n"
    (out_dir / "gap_report.md").write_text(text)
    return text


def run(sows: list = None) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = suite.discover_pairs()
    sows = sows if sows is not None else sows_with_any_signal(pairs)

    reports = [gap_report_for_sow(sow, pairs.get(sow)) for sow in sows]

    long_df = to_long_df(reports)
    summary_df = to_summary_df(reports)
    long_df.to_csv(OUT_DIR / "gap_report_long.csv", index=False)
    summary_df.to_csv(OUT_DIR / "gap_report_summary.csv", index=False)
    report_text = write_report(reports, long_df, summary_df, OUT_DIR)

    print(report_text)
    print(f"Wrote {OUT_DIR}/gap_report_long.csv, gap_report_summary.csv, gap_report.md")
    return {"sows": sows, "reports": reports}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sow", nargs="+", default=None, dest="sows",
                         help="Restrict to specific sows. Defaults to every sow with an "
                              "accelerometer or video file.")
    args = parser.parse_args()
    run(sows=args.sows)
