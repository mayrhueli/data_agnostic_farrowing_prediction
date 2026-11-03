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

"""Tests for named_sow_eval.py: per-instance (not aggregated) predictions
for a chosen list of sows, both farrowing-anchored and online, built from
the same per-row scoring machinery online_eval.py uses for its aggregate
report. Synthetic data throughout except the real-data smoke test, gated
like the rest of this test suite.
"""
import numpy as np
import pandas as pd
import pytest

from video_activity import named_sow_eval, online_eval
from video_activity.ingestion import ACCELEROMETER_DIR, VIDEO_DIR, META_DIR
from nest_building.evaluation import DetectionResult
from farrowing_prediction.training.operon_sr import FormulaModel
from farrowing_prediction.training.instance_tables import FEATURE_COLUMNS
from farrowing_prediction.training.cross_signal import ACTIVITY_FEATURES, fit_activity_scaler

SOW = "sample_sow"
requires_sample_data = pytest.mark.skipif(
    not (ACCELEROMETER_DIR / f"{SOW}.csv").exists()
    or not (VIDEO_DIR / f"{SOW}_activity.csv").exists()
    or not (META_DIR / f"{SOW}.csv").exists(),
    reason="sample sow data not present under data/raw and data/raw/meta",
)


def _flat_formula_model(value_minutes: float = 600.0) -> FormulaModel:
    return FormulaModel(f"({value_minutes} + (0.0 * mean_day))", feature_names=FEATURE_COLUMNS)


def _fake_series(start, n_minutes, value=5.0):
    timestamps = pd.date_range(start, periods=n_minutes, freq="1min")
    return pd.DataFrame({"timestamp": timestamps, "Acc": value})


def _scoreable_table(n=10, sow="A", scale=1.0):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(loc=scale, scale=scale * 0.05, size=(n, len(ACTIVITY_FEATURES))),
                       columns=ACTIVITY_FEATURES)
    df["currently_crated"] = 0
    df["crated_since"] = 0.0
    df["remaining_minutes"] = np.linspace(600, 60, n)
    df["sow"] = sow
    return df


# ---------------------------------------------------------------------------
# anchored_predictions_for_sow
# ---------------------------------------------------------------------------

def test_anchored_predictions_covers_both_signals(monkeypatch):
    series = _fake_series("2020-01-01", 10 * 1440)
    monkeypatch.setitem(named_sow_eval.LOADERS, "accelerometer", lambda sow: series)
    monkeypatch.setitem(named_sow_eval.LOADERS, "video", lambda sow: series)
    monkeypatch.setattr(named_sow_eval, "load_meta", lambda sow: pd.Series({
        "start_farrowing": pd.Timestamp("2020-01-08"), "close_crate": pd.NaT, "open_crate": pd.NaT,
    }))

    scaler = fit_activity_scaler(_scoreable_table(scale=5.0))
    model = _flat_formula_model()

    result = named_sow_eval.anchored_predictions_for_sow("X", model, scaler, {})

    assert set(result["signal"]) == {"accelerometer", "video"}
    assert set(result.loc[result["signal"] == "accelerometer", "scaling_strategy"]) == {"native_accelerometer_scaler"}
    assert set(result.loc[result["signal"] == "video", "scaling_strategy"]) == \
        {"naive_accelerometer_scaler", "video_domain_scaler"}  # no per_sow_baseline entry given


def test_anchored_predictions_isolates_a_failing_signal(monkeypatch):
    good_series = _fake_series("2020-01-01", 10 * 1440)

    def broken_video(sow):
        raise FileNotFoundError("no such file")

    monkeypatch.setitem(named_sow_eval.LOADERS, "accelerometer", lambda sow: good_series)
    monkeypatch.setitem(named_sow_eval.LOADERS, "video", broken_video)
    monkeypatch.setattr(named_sow_eval, "load_meta", lambda sow: pd.Series({
        "start_farrowing": pd.Timestamp("2020-01-08"), "close_crate": pd.NaT, "open_crate": pd.NaT,
    }))

    scaler = fit_activity_scaler(_scoreable_table(scale=5.0))
    result = named_sow_eval.anchored_predictions_for_sow("X", _flat_formula_model(), scaler, {})

    assert set(result["signal"]) == {"accelerometer"}  # video skipped, not crashed


def test_anchored_predictions_empty_when_no_signal_succeeds(monkeypatch):
    def broken(sow):
        raise FileNotFoundError("no such file")

    monkeypatch.setitem(named_sow_eval.LOADERS, "accelerometer", broken)
    monkeypatch.setitem(named_sow_eval.LOADERS, "video", broken)
    monkeypatch.setattr(named_sow_eval, "load_meta", lambda sow: pd.Series({
        "start_farrowing": pd.Timestamp("2020-01-08"), "close_crate": pd.NaT, "open_crate": pd.NaT,
    }))

    result = named_sow_eval.anchored_predictions_for_sow("X", _flat_formula_model(), None, {})
    assert result.empty
    assert list(result.columns) == ["sow", "signal", "scaling_strategy", "y_true_hours", "y_pred_hours"]


# ---------------------------------------------------------------------------
# online_predictions_for_sow
# ---------------------------------------------------------------------------

def test_online_predictions_skips_methods_with_no_window(monkeypatch):
    series = _fake_series("2020-01-01", 10 * 1440)
    farrowing_time = pd.Timestamp("2020-01-08")
    alarm_time = pd.Timestamp("2020-01-06")

    monkeypatch.setitem(named_sow_eval.LOADERS, "accelerometer", lambda sow: series)
    monkeypatch.setitem(named_sow_eval.LOADERS, "video", lambda sow: series)
    monkeypatch.setattr(online_eval, "load_meta", lambda sow: pd.Series({
        "start_farrowing": farrowing_time, "entry_farrowing_pen": pd.Timestamp("2020-01-01"),
        "close_crate": pd.NaT, "open_crate": pd.NaT,
    }))

    fired = DetectionResult("NestDetect", "accelerometer", alarm_time, farrowing_time, "on_time", "on_time", 48.0)

    def fake_detect(df, signal, ft):
        return {"NestDetect": fired, "CUSUM": None, "BEAST": None}

    monkeypatch.setattr(online_eval, "run_detection_methods", fake_detect)

    scaler = fit_activity_scaler(_scoreable_table(scale=5.0))
    result = named_sow_eval.online_predictions_for_sow("X", _flat_formula_model(), scaler, {},
                                                         methods=("NestDetect", "CUSUM"))

    assert set(result["method"]) == {"NestDetect"}  # CUSUM never fired -> no rows, not an error


def test_online_predictions_empty_when_nothing_fires(monkeypatch):
    series = _fake_series("2020-01-01", 10 * 1440)
    monkeypatch.setitem(named_sow_eval.LOADERS, "accelerometer", lambda sow: series)
    monkeypatch.setitem(named_sow_eval.LOADERS, "video", lambda sow: series)
    monkeypatch.setattr(online_eval, "load_meta", lambda sow: pd.Series({
        "start_farrowing": pd.Timestamp("2020-01-08"), "entry_farrowing_pen": pd.Timestamp("2020-01-01"),
        "close_crate": pd.NaT, "open_crate": pd.NaT,
    }))
    monkeypatch.setattr(online_eval, "run_detection_methods",
                         lambda df, signal, ft: {"NestDetect": None, "CUSUM": None, "BEAST": None})

    result = named_sow_eval.online_predictions_for_sow("X", _flat_formula_model(), None, {})
    assert result.empty
    assert list(result.columns) == ["sow", "signal", "method", "scaling_strategy", "y_true_hours", "y_pred_hours"]


# ---------------------------------------------------------------------------
# run(): end-to-end wiring
# ---------------------------------------------------------------------------

def test_run_writes_both_csvs(monkeypatch, tmp_path):
    monkeypatch.setattr(named_sow_eval, "OUT_DIR", tmp_path)
    monkeypatch.setattr(named_sow_eval, "load_selected_model", lambda d: (_flat_formula_model(), None, {}))

    anchored = pd.DataFrame({"sow": ["A"], "signal": ["accelerometer"],
                              "scaling_strategy": ["native_accelerometer_scaler"],
                              "y_true_hours": [1.0], "y_pred_hours": [1.1]})
    online = pd.DataFrame({"sow": ["A"], "signal": ["accelerometer"], "method": ["NestDetect"],
                            "scaling_strategy": ["native_accelerometer_scaler"],
                            "y_true_hours": [1.0], "y_pred_hours": [1.1]})
    monkeypatch.setattr(named_sow_eval, "anchored_predictions_for_sow", lambda *a, **k: anchored)
    monkeypatch.setattr(named_sow_eval, "online_predictions_for_sow", lambda *a, **k: online)

    out = named_sow_eval.run(sows=["A"], sr_transfer_dir=tmp_path)

    assert (tmp_path / "anchored_predictions.csv").exists()
    assert (tmp_path / "online_predictions.csv").exists()
    assert len(out["anchored"]) == 1
    assert len(out["online"]) == 1


# ---------------------------------------------------------------------------
# Real data smoke test
# ---------------------------------------------------------------------------

@requires_sample_data
def test_anchored_predictions_runs_on_real_data():
    scaler = fit_activity_scaler(_scoreable_table(scale=5.0))
    result = named_sow_eval.anchored_predictions_for_sow(SOW, _flat_formula_model(), scaler, {})
    assert not result.empty
    assert (result["sow"] == SOW).all()
