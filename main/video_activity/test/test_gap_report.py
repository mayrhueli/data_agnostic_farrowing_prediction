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

"""Tests for the recording-gap report: gap detection (on both literally-
missing rows and video's resample()-filled NaN runs), farrowing-relative
positioning, per-sow/signal error isolation, and aggregation. Synthetic
data throughout except the real-data smoke test, gated like the rest of
this test suite.
"""
import pandas as pd
import pytest

from video_activity import gap_report
from video_activity.ingestion import ACCELEROMETER_DIR, VIDEO_DIR, META_DIR

SOW = "sample_sow"
requires_sample_data = pytest.mark.skipif(
    not (ACCELEROMETER_DIR / f"{SOW}.csv").exists()
    or not (VIDEO_DIR / f"{SOW}_activity.csv").exists()
    or not (META_DIR / f"{SOW}.csv").exists(),
    reason="sample sow data not present under data/raw and data/raw/meta",
)


def _series(timestamps, acc=None):
    timestamps = pd.to_datetime(timestamps)
    acc = acc if acc is not None else [1.0] * len(timestamps)
    return pd.DataFrame({"timestamp": timestamps, "Acc": acc})


# ---------------------------------------------------------------------------
# detect_gaps: missing-row style (accelerometer) and NaN-run style (video)
# ---------------------------------------------------------------------------

def test_detect_gaps_finds_a_missing_row_gap():
    # 1-minute series with a 2-hour jump in the middle
    before = pd.date_range("2020-01-01 00:00", periods=5, freq="1min")
    after = pd.date_range("2020-01-01 02:04", periods=5, freq="1min")
    series = _series(list(before) + list(after))

    gaps = gap_report.detect_gaps(series)

    assert len(gaps) == 1
    assert gaps.iloc[0]["gap_start"] == before[-1]
    assert gaps.iloc[0]["gap_end"] == after[0]
    assert gaps.iloc[0]["duration_hours"] == pytest.approx(
        (after[0] - before[-1]).total_seconds() / 3600
    )


def test_detect_gaps_finds_a_nan_run_gap():
    # video-style: every minute present, but some have NaN Acc (resample() fill)
    timestamps = pd.date_range("2020-01-01 00:00", periods=200, freq="1min")
    acc = [1.0] * 200
    for i in range(50, 110):  # a 60-minute NaN run
        acc[i] = float("nan")
    series = _series(timestamps, acc)

    gaps = gap_report.detect_gaps(series)

    assert len(gaps) == 1
    assert gaps.iloc[0]["duration_hours"] == pytest.approx(61 / 60, abs=1e-6)  # 49->110 inclusive span + 1min


def test_detect_gaps_ignores_normal_jitter():
    # spacing within tolerance (default 1.5x = 90s) must not be flagged
    timestamps = [pd.Timestamp("2020-01-01 00:00:00"), pd.Timestamp("2020-01-01 00:01:20"),
                  pd.Timestamp("2020-01-01 00:02:35")]
    series = _series(timestamps)
    gaps = gap_report.detect_gaps(series)
    assert gaps.empty


def test_detect_gaps_empty_or_single_row_series():
    assert gap_report.detect_gaps(_series([])).empty
    assert gap_report.detect_gaps(_series(["2020-01-01 00:00"])).empty


# ---------------------------------------------------------------------------
# classify_position
# ---------------------------------------------------------------------------

def test_classify_position_before_after_and_spanning():
    farrowing_time = pd.Timestamp("2020-01-10 12:00:00")
    gaps = pd.DataFrame({
        "gap_start": pd.to_datetime(["2020-01-08 00:00", "2020-01-10 11:00", "2020-01-11 00:00"]),
        "gap_end": pd.to_datetime(["2020-01-08 06:00", "2020-01-10 13:00", "2020-01-11 06:00"]),
        "duration_hours": [6.0, 2.0, 6.0],
    })

    classified = gap_report.classify_position(gaps, farrowing_time)

    assert list(classified["position"]) == ["before farrowing", "spans farrowing", "after farrowing"]
    # positive hours_before_farrowing_start means the gap started before farrowing
    assert classified.iloc[0]["hours_before_farrowing_start"] > 0
    # negative means it started after farrowing
    assert classified.iloc[2]["hours_before_farrowing_start"] < 0


def test_classify_position_handles_empty_gaps():
    farrowing_time = pd.Timestamp("2020-01-10 12:00:00")
    empty = pd.DataFrame(columns=["gap_start", "gap_end", "duration_hours"])
    classified = gap_report.classify_position(empty, farrowing_time)
    assert classified.empty
    assert "position" in classified.columns


# ---------------------------------------------------------------------------
# gap_report_for_sow: per-signal error isolation, missing metadata
# ---------------------------------------------------------------------------

def test_gap_report_for_sow_isolates_a_broken_signal(monkeypatch):
    good_series = _series(pd.date_range("2020-01-01", periods=10, freq="1min"))

    def fake_acc(sow):
        return good_series

    def fake_video(sow):
        raise FileNotFoundError("no such file")

    def fake_meta(sow):
        return pd.Series({"start_farrowing": pd.Timestamp("2020-01-05")})

    monkeypatch.setitem(gap_report.LOADERS, "accelerometer", fake_acc)
    monkeypatch.setitem(gap_report.LOADERS, "video", fake_video)
    monkeypatch.setattr(gap_report, "load_meta", fake_meta)

    report = gap_report.gap_report_for_sow("X")

    assert "error" not in report["accelerometer"]
    assert report["accelerometer"]["n_gaps"] == 0
    assert "error" in report["video"]


def test_gap_report_for_sow_handles_missing_meta(monkeypatch):
    before = pd.date_range("2020-01-01 00:00", periods=5, freq="1min")
    after = pd.date_range("2020-01-01 02:04", periods=5, freq="1min")
    gappy_series = _series(list(before) + list(after))
    monkeypatch.setitem(gap_report.LOADERS, "accelerometer", lambda sow: gappy_series)
    monkeypatch.setitem(gap_report.LOADERS, "video", lambda sow: gappy_series)

    def broken_meta(sow):
        raise ValueError("no metadata file")

    monkeypatch.setattr(gap_report, "load_meta", broken_meta)

    report = gap_report.gap_report_for_sow("X")

    assert report["meta_error"] is not None
    assert report["farrowing_time"] is None
    # gap counts still reported even without metadata
    assert report["accelerometer"]["n_gaps"] == 1
    assert len(report["accelerometer"]["gaps"]) == 1
    # but no farrowing-relative columns, since there's no farrowing_time to position against
    assert "position" not in report["accelerometer"]["gaps"][0]
    assert "hours_before_farrowing_start" not in report["accelerometer"]["gaps"][0]


def test_gap_report_for_sow_skips_signal_the_sow_does_not_have(monkeypatch):
    monkeypatch.setattr(gap_report, "load_meta", lambda sow: pd.Series({"start_farrowing": pd.Timestamp("2020-01-05")}))
    report = gap_report.gap_report_for_sow("X", pairs_flags={"meta": True, "accelerometer": True, "video": False})
    assert "video" not in report


# ---------------------------------------------------------------------------
# Aggregation: to_long_df / to_summary_df
# ---------------------------------------------------------------------------

def _fake_report(sow, n_gaps=2, total_hours=5.0, error_signal=None):
    report = {"sow": sow, "farrowing_time": "2020-01-10", "meta_error": None}
    for signal in ["accelerometer", "video"]:
        if signal == error_signal:
            report[signal] = {"error": "boom"}
        else:
            gaps = [{"gap_start": "t0", "gap_end": "t1", "duration_hours": total_hours / n_gaps,
                     "hours_before_farrowing_start": 10.0, "hours_before_farrowing_end": 9.0,
                     "position": "before farrowing"} for _ in range(n_gaps)]
            report[signal] = {"n_gaps": n_gaps, "total_gap_hours": total_hours,
                               "largest_gap_hours": total_hours / n_gaps, "mean_gap_hours": total_hours / n_gaps,
                               "gaps": gaps}
    return report


def test_to_long_df_pools_gaps_across_sows_and_signals():
    reports = [_fake_report("A", n_gaps=2), _fake_report("B", n_gaps=1, error_signal="video")]
    long_df = gap_report.to_long_df(reports)
    assert len(long_df) == 4 + 1  # A: 2 acc + 2 video rows, B: 1 acc row (video errored, excluded)
    assert set(long_df["sow"]) == {"A", "B"}


def test_to_summary_df_reports_errors_separately():
    reports = [_fake_report("A"), _fake_report("B", error_signal="accelerometer")]
    summary = gap_report.to_summary_df(reports)
    b_acc = summary[(summary["sow"] == "B") & (summary["signal"] == "accelerometer")].iloc[0]
    assert b_acc["error"] == "boom"
    a_acc = summary[(summary["sow"] == "A") & (summary["signal"] == "accelerometer")].iloc[0]
    assert a_acc["n_gaps"] == 2


def test_aggregation_handles_no_reports():
    assert gap_report.to_long_df([]).empty
    assert gap_report.to_summary_df([]).empty


# ---------------------------------------------------------------------------
# run(): end-to-end wiring, output files
# ---------------------------------------------------------------------------

def test_run_writes_csvs_and_report(monkeypatch, tmp_path):
    good_series = _series(pd.date_range("2020-01-01", periods=10, freq="1min"))
    monkeypatch.setattr(gap_report, "OUT_DIR", tmp_path)
    monkeypatch.setitem(gap_report.LOADERS, "accelerometer", lambda sow: good_series)
    monkeypatch.setitem(gap_report.LOADERS, "video", lambda sow: good_series)
    monkeypatch.setattr(gap_report, "load_meta",
                         lambda sow: pd.Series({"start_farrowing": pd.Timestamp("2020-01-05")}))
    monkeypatch.setattr(gap_report.suite, "discover_pairs", lambda: {
        "A": {"meta": True, "accelerometer": True, "video": True},
    })

    out = gap_report.run()

    assert out["sows"] == ["A"]
    assert (tmp_path / "gap_report_long.csv").exists()
    assert (tmp_path / "gap_report_summary.csv").exists()
    assert (tmp_path / "gap_report.md").exists()


def test_run_with_no_sows_does_not_crash(monkeypatch, tmp_path):
    monkeypatch.setattr(gap_report, "OUT_DIR", tmp_path)
    monkeypatch.setattr(gap_report.suite, "discover_pairs", lambda: {})
    out = gap_report.run()
    assert out["sows"] == []
    assert (tmp_path / "gap_report_summary.csv").exists()


# ---------------------------------------------------------------------------
# Real data smoke test
# ---------------------------------------------------------------------------

@requires_sample_data
def test_gap_report_for_sow_runs_on_real_data():
    report = gap_report.gap_report_for_sow(SOW)
    assert report["sow"] == SOW
    assert report["meta_error"] is None
    for signal in ["accelerometer", "video"]:
        assert "error" not in report[signal]
        assert report[signal]["n_gaps"] >= 0
        assert report[signal]["total_gap_hours"] >= 0
