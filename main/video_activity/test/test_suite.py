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

"""Tests for the batch suite: discovery of complete/incomplete pairs,
aggregation math (synthetic, so it's verified for >1 sow even before you
have more than one real pair), and that one bad sow can't take down the
whole run.

The real-data tests are parametrized dynamically over whatever complete
accelerometer+video+meta pairs are found under data/, so running
``pytest`` after dropping in new sows automatically covers them.
"""
import pandas as pd
import pytest

from video_activity import suite
from video_activity.ingestion import ACCELEROMETER_DIR, VIDEO_DIR, META_DIR


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def test_discover_pairs_flags_each_file_type(tmp_path, monkeypatch):
    acc_dir, video_dir, meta_dir = tmp_path / "acc", tmp_path / "video", tmp_path / "meta"
    acc_dir.mkdir(); video_dir.mkdir(); meta_dir.mkdir()

    (acc_dir / "A.csv").write_text("timestamp,Acc\n")
    (video_dir / "A_activity.csv").write_text("Timestamp,Flow_totals\n")
    (meta_dir / "A.csv").write_text("x\n")  # complete triplet

    (acc_dir / "B.csv").write_text("timestamp,Acc\n")
    (meta_dir / "B.csv").write_text("x\n")  # missing video

    (video_dir / "C_activity.csv").write_text("Timestamp,Flow_totals\n")  # missing acc + meta

    monkeypatch.setattr(suite, "ACCELEROMETER_DIR", acc_dir)
    monkeypatch.setattr(suite, "VIDEO_DIR", video_dir)
    monkeypatch.setattr(suite, "META_DIR", meta_dir)

    pairs = suite.discover_pairs()
    assert pairs["A"] == {"meta": True, "accelerometer": True, "video": True}
    assert pairs["B"] == {"meta": True, "accelerometer": True, "video": False}
    assert pairs["C"] == {"meta": False, "accelerometer": False, "video": True}

    assert suite.complete_pairs(pairs) == ["A"]
    missing = suite.incomplete_pairs(pairs)
    assert missing["B"] == ["video"]
    assert set(missing["C"]) == {"meta", "accelerometer"}


# ---------------------------------------------------------------------------
# Aggregation math (synthetic summaries, mirroring case_study.main()'s output)
# ---------------------------------------------------------------------------

def _fake_summary(sow, nestdetect_cls_acc, nestdetect_cls_video):
    def det(cls, lead=10.0):
        return {"changepoint": "2020-01-01", "lead_time_hours": lead,
                "classification_24h": cls, "classification_48h": cls}

    return {
        "sow": sow,
        "detection": {
            "accelerometer": {"NestDetect": det(nestdetect_cls_acc), "CUSUM": None, "BEAST": None},
            "video": {"NestDetect": det(nestdetect_cls_video), "CUSUM": None, "BEAST": None},
        },
    }


def test_detection_stats_aggregates_across_sows():
    results = {
        "sow1": _fake_summary("sow1", "on_time", "on_time"),
        "sow2": _fake_summary("sow2", "on_time", "early"),
        "sow3": _fake_summary("sow3", "late", "early"),
    }
    long_df = suite.detection_long(results)
    assert len(long_df) == 3 * 2  # 3 sows x 2 signals (NestDetect only, CUSUM/BEAST are None)

    stats = suite.detection_stats(long_df)
    nd_acc = stats[(stats.method == "NestDetect") & (stats.signal == "accelerometer")].iloc[0]
    assert nd_acc["n"] == 3
    assert nd_acc["on_time_48h_pct"] == pytest.approx(200 / 3, abs=0.1)
    assert nd_acc["late_48h_pct"] == pytest.approx(100 / 3, abs=0.1)

    nd_video = stats[(stats.method == "NestDetect") & (stats.signal == "video")].iloc[0]
    assert nd_video["on_time_48h_pct"] == pytest.approx(100 / 3, abs=0.1)
    assert nd_video["early_48h_pct"] == pytest.approx(200 / 3, abs=0.1)


def test_aggregation_handles_no_results():
    assert suite.detection_stats(suite.detection_long({})).empty


def test_report_falls_back_when_to_markdown_is_unavailable(monkeypatch):
    """A stale `tabulate` install (seen with Anaconda's default, 0.8.10 —
    pandas.to_markdown needs >=0.9.0) raises ImportError deep inside
    pandas; the report must still render, just as a plain table, rather
    than losing the whole suite run over a reporting nicety.
    """
    df = pd.DataFrame({"method": ["CUSUM"], "signal": ["accelerometer"], "n": [3]})

    def broken_to_markdown(self, *args, **kwargs):
        raise ImportError("Pandas requires version '0.9.0' or newer of 'tabulate'")

    monkeypatch.setattr(pd.DataFrame, "to_markdown", broken_to_markdown)

    rendered = suite._df_to_markdown_or_text(df)
    assert "CUSUM" in rendered
    assert "```" in rendered  # fenced fallback, not a crash


# ---------------------------------------------------------------------------
# Error isolation: one bad sow must not stop the rest
# ---------------------------------------------------------------------------

def test_run_suite_isolates_a_failing_sow(monkeypatch, tmp_path):
    def fake_main(sow):
        if sow == "bad":
            raise ValueError("boom")
        return _fake_summary(sow, "on_time", "on_time")

    monkeypatch.setattr(suite.case_study, "main", fake_main)
    monkeypatch.setattr(suite, "SUITE_DIR", tmp_path)
    monkeypatch.setattr(suite.case_study, "OUT_DIR", tmp_path)

    out = suite.run_suite(sows=["good1", "bad", "good2"])

    assert set(out["sows"]) == {"good1", "good2"}
    assert list(out["errors"].keys()) == ["bad"]
    assert "boom" in out["errors"]["bad"]
    assert (tmp_path / "suite_report.md").exists()
    assert (tmp_path / "detection_summary.csv").exists()


def test_run_suite_with_no_complete_pairs_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(suite, "ACCELEROMETER_DIR", tmp_path / "empty_acc")
    monkeypatch.setattr(suite, "VIDEO_DIR", tmp_path / "empty_video")
    monkeypatch.setattr(suite, "META_DIR", tmp_path / "empty_meta")
    (tmp_path / "empty_acc").mkdir()
    (tmp_path / "empty_video").mkdir()
    (tmp_path / "empty_meta").mkdir()

    out = suite.run_suite()
    assert out["sows"] == []


# ---------------------------------------------------------------------------
# Real data: parametrized dynamically over whatever pairs you actually have
# ---------------------------------------------------------------------------

def _real_complete_pairs():
    try:
        return suite.complete_pairs()
    except Exception:
        return []


@pytest.mark.parametrize("sow", _real_complete_pairs())
def test_case_study_runs_for_every_pair_you_have(sow):
    from video_activity import case_study
    summary = case_study.main(sow)
    assert summary["sow"] == sow
    for signal in ("accelerometer", "video"):
        assert signal in summary["detection"]
