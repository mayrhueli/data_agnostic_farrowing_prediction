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

"""Tests for the online evaluation pipeline: detection-triggered,
walk-forward prediction using the model sr_transfer.py actually
persisted (not a retrained stand-in), for both accelerometer and video.
Synthetic data throughout except the real-data smoke test, gated like
the rest of this test suite.
"""
import json

import numpy as np
import pandas as pd
import pytest

from video_activity import online_eval
from video_activity.ingestion import ACCELEROMETER_DIR, VIDEO_DIR, META_DIR
from nest_building.evaluation import DetectionResult
from farrowing_prediction.training.operon_sr import FormulaModel
from farrowing_prediction.training.cross_signal import ACTIVITY_FEATURES
from farrowing_prediction.training.instance_tables import FEATURE_COLUMNS

SOW = "sample_sow"
requires_sample_data = pytest.mark.skipif(
    not (ACCELEROMETER_DIR / f"{SOW}.csv").exists()
    or not (VIDEO_DIR / f"{SOW}_activity.csv").exists()
    or not (META_DIR / f"{SOW}.csv").exists(),
    reason="sample sow data not present under data/raw and data/raw/meta",
)


def _flat_formula_model(value_minutes: float = 600.0) -> FormulaModel:
    """A trivial model that always predicts the same remaining_minutes,
    regardless of input — enough to exercise scoring without caring
    about actual accuracy."""
    return FormulaModel(f"({value_minutes} + (0.0 * mean_day))", feature_names=FEATURE_COLUMNS)


def _fake_series(start, n_minutes, value=1.0):
    timestamps = pd.date_range(start, periods=n_minutes, freq="1min")
    return pd.DataFrame({"timestamp": timestamps, "Acc": value})


# ---------------------------------------------------------------------------
# load_selected_model / accelerometer_training_pool_sows
# ---------------------------------------------------------------------------

def test_load_selected_model_raises_clearly_when_sr_transfer_not_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="sr_transfer"):
        online_eval.load_selected_model(tmp_path)


def test_load_selected_model_round_trips(tmp_path):
    from farrowing_prediction.training.operon_sr import OperonSymbolicRegressorWrapper
    from sklearn.preprocessing import StandardScaler
    from joblib import dump

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(30, 3)), columns=["a", "b", "c"])
    y = 2 * X["a"] + 5
    model = OperonSymbolicRegressorWrapper(population_size=50, generations=10, n_threads=1, random_state=0)
    model.fit(X, y)
    model.dump_model(str(tmp_path / "model.json"))
    dump(StandardScaler().fit(X), tmp_path / "accelerometer_scaler.joblib")
    (tmp_path / "per_sow_baselines.json").write_text(json.dumps({"sow_a": 12.5}))

    loaded_model, scaler, baselines = online_eval.load_selected_model(tmp_path)
    assert isinstance(loaded_model, FormulaModel)
    assert baselines == {"sow_a": 12.5}
    assert hasattr(scaler, "transform")


def test_load_selected_model_defaults_baselines_to_empty_dict_when_missing(tmp_path):
    from farrowing_prediction.training.operon_sr import OperonSymbolicRegressorWrapper
    from sklearn.preprocessing import StandardScaler
    from joblib import dump

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(30, 3)), columns=["a", "b", "c"])
    model = OperonSymbolicRegressorWrapper(population_size=50, generations=10, n_threads=1, random_state=0)
    model.fit(X, 2 * X["a"] + 5)
    model.dump_model(str(tmp_path / "model.json"))
    dump(StandardScaler().fit(X), tmp_path / "accelerometer_scaler.joblib")
    # no per_sow_baselines.json written

    _, _, baselines = online_eval.load_selected_model(tmp_path)
    assert baselines == {}


def test_accelerometer_training_pool_sows_derives_best_fold_minus_its_test_sows(tmp_path):
    oof = pd.DataFrame({
        "sow": ["A", "A", "B", "B", "C", "C"],
        "fold": [0, 0, 0, 0, 1, 1],
        "y_true_hours": [1, 2, 3, 4, 5, 6],
        "y_pred_hours": [1, 2, 3, 4, 5, 6],
    })
    oof.to_csv(tmp_path / "accelerometer_cv_oof.csv", index=False)
    summary = {"cv_folds": [{"fold": 0, "mae_hours": 1.0}, {"fold": 1, "mae_hours": 5.0}]}
    (tmp_path / "sr_transfer_summary.json").write_text(json.dumps(summary))

    pool = online_eval.accelerometer_training_pool_sows(tmp_path)
    # best fold (lowest mae) is fold 0, tested on A and B -> those are
    # NOT in that model's training data; only C (fold 1, not the best) is
    assert pool == {"C"}


def test_accelerometer_training_pool_sows_empty_when_files_missing(tmp_path):
    assert online_eval.accelerometer_training_pool_sows(tmp_path) == set()


# ---------------------------------------------------------------------------
# _scaled_predictions
# ---------------------------------------------------------------------------

def _scoreable_table(n=10, sow="A", scale=1.0):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(loc=scale, scale=scale * 0.05, size=(n, len(ACTIVITY_FEATURES))),
                       columns=ACTIVITY_FEATURES)
    df["currently_crated"] = 0
    df["crated_since"] = 0.0
    df["remaining_minutes"] = np.linspace(600, 60, n)
    df["sow"] = sow
    return df


def test_scaled_predictions_accelerometer_uses_only_native_scaler():
    from farrowing_prediction.training.cross_signal import fit_activity_scaler
    table = _scoreable_table(scale=5.0)
    scaler = fit_activity_scaler(table)
    model = _flat_formula_model()

    results = online_eval._scaled_predictions(table, "accelerometer", "A", model, scaler, {})
    assert set(results) == {"native_accelerometer_scaler"}
    assert results["native_accelerometer_scaler"]["n_instances"] == len(table)


def test_scaled_predictions_video_includes_per_sow_only_when_baseline_available():
    from farrowing_prediction.training.cross_signal import fit_activity_scaler
    table = _scoreable_table(scale=5.0 * 1e5, sow="A")
    accelerometer_scaler = fit_activity_scaler(_scoreable_table(scale=5.0))
    model = _flat_formula_model()

    without_baseline = online_eval._scaled_predictions(table, "video", "A", model, accelerometer_scaler, {})
    assert set(without_baseline) == {"naive_accelerometer_scaler", "video_domain_scaler"}

    with_baseline = online_eval._scaled_predictions(
        table, "video", "A", model, accelerometer_scaler, {"A": 5.0 * 1e5})
    assert set(with_baseline) == {"naive_accelerometer_scaler", "video_domain_scaler", "per_sow_baseline"}


# ---------------------------------------------------------------------------
# evaluate_online_for_sow_signal: the detection -> online-window wiring
# ---------------------------------------------------------------------------

def test_evaluate_online_reports_no_window_when_detector_never_fires(monkeypatch):
    series = _fake_series("2020-01-01", 10 * 1440)

    monkeypatch.setitem(online_eval.LOADERS, "accelerometer", lambda sow: series)
    monkeypatch.setattr(online_eval, "load_meta", lambda sow: pd.Series({
        "start_farrowing": pd.Timestamp("2020-01-08"),
        "entry_farrowing_pen": pd.Timestamp("2020-01-01"),
        "close_crate": pd.NaT, "open_crate": pd.NaT,
    }))
    monkeypatch.setattr(online_eval, "run_detection_methods", lambda df, signal, farrowing_time: {
        "NestDetect": None, "CUSUM": None, "BEAST": None,
    })

    rows = online_eval.evaluate_online_for_sow_signal("X", "accelerometer", _flat_formula_model(),
                                                        None, {}, methods=("NestDetect",))
    assert len(rows) == 1
    assert rows[0]["online_eval"] is None
    assert "did not fire" in rows[0]["online_eval_error"]


def test_evaluate_online_reports_no_window_when_detector_fires_after_farrowing(monkeypatch):
    series = _fake_series("2020-01-01", 10 * 1440)
    farrowing_time = pd.Timestamp("2020-01-08")

    monkeypatch.setitem(online_eval.LOADERS, "accelerometer", lambda sow: series)
    monkeypatch.setattr(online_eval, "load_meta", lambda sow: pd.Series({
        "start_farrowing": farrowing_time, "entry_farrowing_pen": pd.Timestamp("2020-01-01"),
        "close_crate": pd.NaT, "open_crate": pd.NaT,
    }))
    late_detection = DetectionResult("NestDetect", "accelerometer", farrowing_time + pd.Timedelta(hours=1),
                                      farrowing_time, "late", "late", None)
    monkeypatch.setattr(online_eval, "run_detection_methods", lambda df, signal, ft: {
        "NestDetect": late_detection, "CUSUM": None, "BEAST": None,
    })

    rows = online_eval.evaluate_online_for_sow_signal("X", "accelerometer", _flat_formula_model(),
                                                        None, {}, methods=("NestDetect",))
    assert rows[0]["online_eval"] is None
    assert "fired at/after farrowing" in rows[0]["online_eval_error"]


def test_evaluate_online_scores_a_valid_window(monkeypatch):
    series = _fake_series("2020-01-01", 10 * 1440, value=5.0)
    farrowing_time = pd.Timestamp("2020-01-08")
    alarm_time = pd.Timestamp("2020-01-06")  # 2 days before farrowing, well after 48h warmup

    monkeypatch.setitem(online_eval.LOADERS, "accelerometer", lambda sow: series)
    monkeypatch.setattr(online_eval, "load_meta", lambda sow: pd.Series({
        "start_farrowing": farrowing_time, "entry_farrowing_pen": pd.Timestamp("2020-01-01"),
        "close_crate": pd.NaT, "open_crate": pd.NaT,
    }))
    detection = DetectionResult("NestDetect", "accelerometer", alarm_time, farrowing_time,
                                 "on_time", "on_time", 48.0)
    monkeypatch.setattr(online_eval, "run_detection_methods", lambda df, signal, ft: {
        "NestDetect": detection, "CUSUM": None, "BEAST": None,
    })

    from farrowing_prediction.training.cross_signal import fit_activity_scaler
    scaler = fit_activity_scaler(_scoreable_table(scale=5.0))
    model = _flat_formula_model(value_minutes=120.0)

    rows = online_eval.evaluate_online_for_sow_signal("X", "accelerometer", model, scaler, {},
                                                        methods=("NestDetect",))
    assert len(rows) == 1
    assert rows[0]["online_eval_error"] is None
    assert rows[0]["online_eval"]["native_accelerometer_scaler"]["n_instances"] > 0
    assert rows[0]["online_eval"]["native_accelerometer_scaler"]["mae_hours"] >= 0


# ---------------------------------------------------------------------------
# to_long_df
# ---------------------------------------------------------------------------

def test_to_long_df_flattens_scaling_strategies_into_rows():
    rows = [
        {"sow": "A", "signal": "video", "method": "NestDetect", "changepoint": "t", "lead_time_hours": 1.0,
         "classification_24h": "on_time", "classification_48h": "on_time",
         "online_eval": {"video_domain_scaler": {"n_instances": 5, "mae_hours": 2.0, "pcc": 0.5}},
         "online_eval_error": None},
        {"sow": "B", "signal": "video", "method": "CUSUM", "changepoint": None, "lead_time_hours": None,
         "classification_24h": None, "classification_48h": None,
         "online_eval": None, "online_eval_error": "detector did not fire"},
    ]
    long_df = online_eval.to_long_df(rows, training_pool_sows=set())
    assert len(long_df) == 2
    scored = long_df[long_df["sow"] == "A"].iloc[0]
    assert scored["scaling_strategy"] == "video_domain_scaler"
    assert scored["mae_hours"] == 2.0
    failed = long_df[long_df["sow"] == "B"].iloc[0]
    assert pd.isna(failed["scaling_strategy"])
    assert pd.isna(failed["mae_hours"])


def test_to_long_df_held_out_reflects_training_pool():
    rows = [
        {"sow": "A", "signal": "accelerometer", "method": "NestDetect", "changepoint": "t",
         "lead_time_hours": 1.0, "classification_24h": "on_time", "classification_48h": "on_time",
         "online_eval": {"native_accelerometer_scaler": {"n_instances": 5, "mae_hours": 2.0, "pcc": 0.5}},
         "online_eval_error": None},
    ]
    long_df = online_eval.to_long_df(rows, training_pool_sows={"A"})
    assert long_df.iloc[0]["held_out"] == False  # noqa: E712 - explicit bool check, not truthiness

    long_df_unknown = online_eval.to_long_df(rows, training_pool_sows=set())
    assert long_df_unknown.iloc[0]["held_out"] is None


# ---------------------------------------------------------------------------
# run(): end-to-end wiring
# ---------------------------------------------------------------------------

def test_run_writes_csv_and_report(monkeypatch, tmp_path):
    out_dir = tmp_path / "out"
    monkeypatch.setattr(online_eval, "OUT_DIR", out_dir)
    monkeypatch.setattr(online_eval, "load_selected_model", lambda d: (_flat_formula_model(), None, {}))
    monkeypatch.setattr(online_eval, "accelerometer_training_pool_sows", lambda d: set())
    monkeypatch.setattr(online_eval.suite, "discover_pairs", lambda: {
        "A": {"meta": True, "accelerometer": True, "video": False},
    })
    monkeypatch.setattr(online_eval, "evaluate_online_for_sow_signal", lambda *a, **k: [
        {"sow": "A", "signal": "accelerometer", "method": "NestDetect", "changepoint": None,
         "lead_time_hours": None, "classification_24h": None, "classification_48h": None,
         "online_eval": None, "online_eval_error": "detector did not fire — no online prediction window"},
    ])

    out = online_eval.run(sows=["A"], sr_transfer_dir=tmp_path)

    assert out["sows"] == ["A"]
    assert (out_dir / "online_eval_long.csv").exists()
    assert (out_dir / "online_eval_report.md").exists()


def test_run_isolates_a_failing_sow(monkeypatch, tmp_path):
    out_dir = tmp_path / "out"
    monkeypatch.setattr(online_eval, "OUT_DIR", out_dir)
    monkeypatch.setattr(online_eval, "load_selected_model", lambda d: (_flat_formula_model(), None, {}))
    monkeypatch.setattr(online_eval, "accelerometer_training_pool_sows", lambda d: set())
    monkeypatch.setattr(online_eval.suite, "discover_pairs", lambda: {
        "good": {"meta": True, "accelerometer": True, "video": False},
        "bad": {"meta": True, "accelerometer": True, "video": False},
    })

    def fake_eval(sow, signal_name, *a, **k):
        if sow == "bad":
            raise ValueError("boom")
        return [{"sow": sow, "signal": signal_name, "method": "NestDetect", "changepoint": None,
                  "lead_time_hours": None, "classification_24h": None, "classification_48h": None,
                  "online_eval": None, "online_eval_error": "detector did not fire"}]

    monkeypatch.setattr(online_eval, "evaluate_online_for_sow_signal", fake_eval)

    out = online_eval.run(sows=["good", "bad"], sr_transfer_dir=tmp_path)
    long_df = out["long_df"]
    assert "boom" in long_df[long_df["sow"] == "bad"]["online_eval_error"].iloc[0]
    assert (long_df["sow"] == "good").any()


# ---------------------------------------------------------------------------
# Real data smoke test: the actual full pipeline, no mocking
# ---------------------------------------------------------------------------

@requires_sample_data
def test_evaluate_online_runs_on_real_data_for_at_least_one_method():
    model = _flat_formula_model(value_minutes=600.0)
    from farrowing_prediction.training.cross_signal import fit_activity_scaler
    scaler = fit_activity_scaler(_scoreable_table(scale=5.0))

    rows = online_eval.evaluate_online_for_sow_signal(SOW, "accelerometer", model, scaler, {},
                                                        methods=("NestDetect", "CUSUM"))
    assert len(rows) == 2
    for row in rows:
        assert row["sow"] == SOW
        # either a window was built and scored, or a clear reason is given — never silently missing
        assert row["online_eval"] is not None or row["online_eval_error"] is not None
