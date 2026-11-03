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

"""Smoke tests for the video-vs-accelerometer case study. Skipped if the
raw data isn't present locally (data/ is gitignored, not part of the repo
checkout), so this only runs where the sample sow has been placed under
data/raw/ as documented in the top-level README.
"""
import pandas as pd
import pytest

from video_activity.ingestion import ACCELEROMETER_DIR, VIDEO_DIR, META_DIR, \
    load_accelerometer_series, load_video_series, load_meta
from nest_building.nestdetect import detect_nest_building
from nest_building.cusum import detect_cusum
from farrowing_prediction.training import instance_tables
from farrowing_prediction.training.instance_tables import build_instance_table, build_online_instance_table

SOW = "sample_sow"

requires_sample_data = pytest.mark.skipif(
    not (ACCELEROMETER_DIR / f"{SOW}.csv").exists()
    or not (VIDEO_DIR / f"{SOW}_activity.csv").exists()
    or not (META_DIR / f"{SOW}.csv").exists(),
    reason="sample sow data not present under data/raw and data/raw/meta",
)


def test_load_meta_treats_uncrated_sentinel_as_never_crated(tmp_path):
    """closing.crate/opening.crate hold the literal string "uncrated" for
    sows that were never crated, not blank/NaN — pd.to_datetime used to
    raise DateParseError on that (hit on ~1/4 of a real 177-sow archive).
    """
    meta_csv = tmp_path / "147999 1.csv"
    meta_csv.write_text(
        '"Date.of.sow.introduction.to.farrowing.pen","Datetime.of.sows.farrowing",'
        '"Sow.number.trial.number","Smartbow.tag.number","pen.type","end.of.farrowing",'
        '"closing.crate","opening.crate"\n'
        '2016-01-15,"2016-01-21 16:41:00","147999 1",33,"F","2016-01-21 20:08:00",'
        '"uncrated","uncrated"\n'
    )

    meta = load_meta("147999 1", base_dir=tmp_path)

    assert meta["sow"] == "147999 1"
    assert pd.isna(meta["close_crate"])
    assert pd.isna(meta["open_crate"])
    assert isinstance(meta["start_farrowing"], pd.Timestamp)


@requires_sample_data
def test_ingestion_produces_1min_acc_series():
    acc = load_accelerometer_series(SOW)
    video = load_video_series(SOW)
    for df in (acc, video):
        assert list(df.columns[:2]) == ["timestamp", "Acc"]
        assert df["Acc"].notna().any()
        assert df["timestamp"].is_monotonic_increasing

    meta = load_meta(SOW)
    assert meta["sow"] == SOW
    assert isinstance(meta["start_farrowing"], pd.Timestamp)


@requires_sample_data
def test_nest_building_detectors_run_without_error():
    acc = load_accelerometer_series(SOW)
    meta = load_meta(SOW)
    window = acc[(acc["timestamp"] >= meta["entry_farrowing_pen"]) &
                 (acc["timestamp"] <= meta["start_farrowing"] + pd.Timedelta(hours=12))].reset_index(drop=True)

    cp_nd, label, _, _ = detect_nest_building(window)
    assert label in {"Alarm 1", "Alarm 2", "Alarm 3", None}

    cp_cusum, s_plus, s_minus, rolling = detect_cusum(window)
    assert len(s_plus) == len(window)
    if cp_cusum is not None:
        assert meta["entry_farrowing_pen"] <= cp_cusum <= meta["start_farrowing"] + pd.Timedelta(hours=12)


# ---------------------------------------------------------------------------
# Regression tests for two crash modes a real 19-sow batch run surfaced:
#   "Found array with 0 sample(s) ... required by StandardScaler" and
#   "Series does not reach the farrowing timestamp" propagating out of
# case_study.main() and killing the whole sow (and its already-computed
# detection results) instead of being isolated. Synthetic, no real data
# needed.
# ---------------------------------------------------------------------------

def test_build_instance_table_survives_allnan_calendar_features(monkeypatch):
    """calculate_features_for_timeseries's mean_day/mean_night are computed
    over a fixed calendar-day window near the start of the recording; for a
    short or gappy series that window can be empty for *every* extracted
    row. This used to leave the table all-NaN there, which dropna() then
    reduced to zero rows -> StandardScaler's cryptic "0 sample(s)" error.
    """
    def fake_features(series, eventtime, full_features=False):
        remaining = pd.Timedelta(eventtime - series["timestamp"].max()).total_seconds() / 60
        return pd.DataFrame([{
            "mean_day": float("nan"), "mean_night": float("nan"),
            "mean_24": 1.0, "mean_48": 1.0,
            "mean_last_12": 1.0, "mean_last_24": 1.0,
            "trend_last_12": 0.0, "trend_last_24": 0.0,
            "remaining_minutes": remaining,
        }])

    monkeypatch.setattr(instance_tables, "calculate_features_for_timeseries", fake_features)

    start = pd.Timestamp("2020-01-01")
    farrowing = start + pd.Timedelta(days=5)
    timestamps = pd.date_range(start, periods=6 * 1440, freq="1min")
    df = pd.DataFrame({"timestamp": timestamps, "Acc": 1.0})

    table = build_instance_table(df, farrowing)
    assert len(table) > 0
    assert table[instance_tables.FEATURE_COLUMNS].isna().sum().sum() == 0


def test_build_instance_table_raises_clear_error_when_too_short():
    """A recording that never accumulates 48h of history before farrowing
    used to raise a bare pandas "No objects to concatenate" from an empty
    `rows` list; it should now say clearly what's wrong.
    """
    start = pd.Timestamp("2020-01-01")
    farrowing = start + pd.Timedelta(hours=30)  # far short of the 48h lookback
    timestamps = pd.date_range(start, periods=32 * 60, freq="1min")  # reaches past farrowing
    df = pd.DataFrame({"timestamp": timestamps, "Acc": 1.0})

    with pytest.raises(ValueError, match="not enough pre-farrowing data"):
        build_instance_table(df, farrowing)


def test_build_instance_table_raises_clear_error_when_series_ends_before_farrowing():
    start = pd.Timestamp("2020-01-01")
    timestamps = pd.date_range(start, periods=3000, freq="1min")
    df = pd.DataFrame({"timestamp": timestamps, "Acc": 1.0})
    farrowing = timestamps.max() + pd.Timedelta(hours=1)  # sensor stopped before farrowing

    with pytest.raises(ValueError, match="does not reach the farrowing timestamp"):
        build_instance_table(df, farrowing)


# ---------------------------------------------------------------------------
# build_online_instance_table: the "online" setting (window starts at a
# detector's alarm time, not a fixed offset before the known farrowing time).
# ---------------------------------------------------------------------------

def test_build_online_instance_table_windows_from_start_time_not_farrowing():
    """Instances should only be built from start_time to farrowing_time —
    nothing before start_time, nothing at/after farrowing_time."""
    start = pd.Timestamp("2020-01-01")
    farrowing = start + pd.Timedelta(days=6)
    timestamps = pd.date_range(start, periods=7 * 1440, freq="1min")
    df = pd.DataFrame({"timestamp": timestamps, "Acc": 1.0})

    alarm_time = start + pd.Timedelta(days=4)  # detector fired 2 days before farrowing
    table = build_online_instance_table(df, alarm_time, farrowing)

    assert len(table) > 0
    # remaining_minutes must stay positive (strictly before farrowing) and
    # never exceed the time from the alarm to farrowing — up to the one
    # extra minute from the first window ending just before alarm_time
    # itself (df.iloc[:i] excludes row i), same off-by-one as
    # build_instance_table's equivalent window-end behavior.
    max_possible = (farrowing - alarm_time).total_seconds() / 60 + 1
    assert (table["remaining_minutes"] > 0).all()
    assert (table["remaining_minutes"] <= max_possible).all()


def test_build_online_instance_table_raises_when_alarm_fires_after_farrowing():
    start = pd.Timestamp("2020-01-01")
    farrowing = start + pd.Timedelta(days=5)
    timestamps = pd.date_range(start, periods=6 * 1440, freq="1min")
    df = pd.DataFrame({"timestamp": timestamps, "Acc": 1.0})

    late_alarm = farrowing + pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="no online prediction window exists"):
        build_online_instance_table(df, late_alarm, farrowing)


def test_build_online_instance_table_raises_when_alarm_equals_farrowing():
    start = pd.Timestamp("2020-01-01")
    farrowing = start + pd.Timedelta(days=5)
    timestamps = pd.date_range(start, periods=6 * 1440, freq="1min")
    df = pd.DataFrame({"timestamp": timestamps, "Acc": 1.0})

    with pytest.raises(ValueError, match="no online prediction window exists"):
        build_online_instance_table(df, farrowing, farrowing)


def test_build_online_instance_table_raises_when_not_enough_history_since_alarm():
    """A recording that never accumulates 48h of history before farrowing
    (regardless of alarm time — even a very early alarm doesn't help,
    since every window still needs TWO_DAYS of history behind it)."""
    start = pd.Timestamp("2020-01-01")
    farrowing = start + pd.Timedelta(hours=30)  # far short of the 48h lookback, like
    timestamps = pd.date_range(start, periods=32 * 60, freq="1min")  # the equivalent build_instance_table test
    df = pd.DataFrame({"timestamp": timestamps, "Acc": 1.0})

    alarm_time = start + pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="not enough history since the alarm fired"):
        build_online_instance_table(df, alarm_time, farrowing)


def test_build_online_instance_table_matches_build_instance_table_when_start_is_days_before():
    """When start_time happens to equal exactly `days_before` days before
    farrowing, the online window should produce the same instances as the
    farrowing-anchored one — the two functions differ only in how the
    window bounds are chosen, not in the per-row feature extraction."""
    start = pd.Timestamp("2020-01-01")
    farrowing = start + pd.Timedelta(days=6)
    timestamps = pd.date_range(start, periods=7 * 1440, freq="1min")
    df = pd.DataFrame({"timestamp": timestamps, "Acc": 1.0})

    anchored = build_instance_table(df, farrowing, days_before=2)
    online = build_online_instance_table(df, farrowing - pd.Timedelta(days=2), farrowing)

    pd.testing.assert_frame_equal(anchored, online)


def test_case_study_detection_isolates_a_method_failure(monkeypatch, tmp_path):
    """One detector raising (e.g. a pathological signal that breaks
    NestDetect specifically) must not cost the other detectors' results
    for either signal — run_detection_methods already isolates per
    method/signal (see its own try/except), this just checks main()
    doesn't lose that isolation end to end.
    """
    from video_activity import case_study

    monkeypatch.setattr(case_study, "OUT_DIR", tmp_path)

    start = pd.Timestamp("2020-01-01")
    farrowing = start + pd.Timedelta(days=5)
    timestamps = pd.date_range(start, periods=6 * 1440, freq="1min")
    acc = pd.DataFrame({"timestamp": timestamps, "Acc": 1.0, "source": "accelerometer"})
    video = pd.DataFrame({"timestamp": timestamps, "Acc": 1.0, "source": "video"})
    meta = pd.Series({
        "sow": "synthetic", "entry_farrowing_pen": start, "start_farrowing": farrowing,
        "end_farrowing": farrowing + pd.Timedelta(hours=3), "pen_type": "F",
        "close_crate": pd.NaT, "open_crate": pd.NaT, "smartbow_tag": 1,
    })

    monkeypatch.setattr(case_study, "load_accelerometer_series", lambda sow: acc)
    monkeypatch.setattr(case_study, "load_video_series", lambda sow: video)
    monkeypatch.setattr(case_study, "load_meta", lambda sow: meta)

    def broken_nestdetect(df):
        raise ValueError("synthetic NestDetect failure")

    monkeypatch.setattr(case_study, "run_nestdetect", broken_nestdetect)

    summary = case_study.main("synthetic")

    for signal in ("accelerometer", "video"):
        assert summary["detection"][signal]["NestDetect"] is None
        assert summary["detection"][signal]["CUSUM"] is not None
