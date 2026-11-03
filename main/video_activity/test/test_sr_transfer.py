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

"""Tests for the Operon symbolic-regression cross-signal pipeline: the
domain-scaling split (activity vs. meta features), grouped cross-sow CV,
and naive-vs-domain-adapted transfer evaluation. Synthetic data throughout
except where real sample data is required (gated, like the rest of this
test suite).
"""
import numpy as np
import pandas as pd
import pytest

from farrowing_prediction.training.instance_tables import FEATURE_COLUMNS
from farrowing_prediction.training.cross_signal import (
    ACTIVITY_FEATURES, META_FEATURES, fit_activity_scaler, apply_scaler,
    grouped_cv_train_accelerometer, evaluate_transfer, apply_per_sow_baseline,
    compute_feature_sensitivity,
)
from farrowing_prediction.training.operon_sr import OperonSymbolicRegressorWrapper
from video_activity.ingestion import ACCELEROMETER_DIR, VIDEO_DIR, META_DIR
from video_activity.pooling import build_pooled_table, per_sow_baseline, compute_per_sow_baselines
from video_activity import sr_transfer
from video_activity.sr_transfer import accelerometer_sows

SOW = "sample_sow"
requires_sample_data = pytest.mark.skipif(
    not (ACCELEROMETER_DIR / f"{SOW}.csv").exists()
    or not (VIDEO_DIR / f"{SOW}_activity.csv").exists()
    or not (META_DIR / f"{SOW}.csv").exists(),
    reason="sample sow data not present under data/raw and data/raw/meta",
)

FAST_SR_KWARGS = dict(population_size=50, generations=10, n_threads=1)


def _synthetic_table(n_sows: int, rows_per_sow: int, scale: float, seed: int = 0) -> pd.DataFrame:
    """A pooled instance table with the right shape/columns but fabricated
    values, at a chosen activity-magnitude scale (to mimic accelerometer
    vs. video's very different units)."""
    rng = np.random.default_rng(seed)
    frames = []
    for s in range(n_sows):
        activity = rng.normal(loc=scale, scale=scale * 0.1, size=(rows_per_sow, len(ACTIVITY_FEATURES)))
        remaining = np.linspace(2880, 60, rows_per_sow)  # 48h down to 1h, like a real pre-farrowing window
        df = pd.DataFrame(activity, columns=ACTIVITY_FEATURES)
        df["currently_crated"] = 0
        df["crated_since"] = 0.0
        df["remaining_minutes"] = remaining
        df["sow"] = f"synthetic_{s}"
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def test_feature_split_covers_all_columns_without_overlap():
    assert set(ACTIVITY_FEATURES) | set(META_FEATURES) == set(FEATURE_COLUMNS)
    assert set(ACTIVITY_FEATURES).isdisjoint(META_FEATURES)


def test_scaler_only_touches_activity_features():
    table = _synthetic_table(n_sows=1, rows_per_sow=20, scale=1000.0)
    table["crated_since"] = np.linspace(0, 500, len(table))  # give it real variance too

    scaler = fit_activity_scaler(table)
    scaled = apply_scaler(table, scaler)

    # activity features standardized
    assert np.allclose(scaled[ACTIVITY_FEATURES].mean(), 0, atol=1e-6)
    assert np.allclose(scaled[ACTIVITY_FEATURES].std(ddof=0), 1, atol=1e-6)
    # meta features passed through completely unchanged
    pd.testing.assert_series_equal(scaled["crated_since"], table["crated_since"])
    pd.testing.assert_series_equal(scaled["currently_crated"], table["currently_crated"])


def test_grouped_cv_requires_at_least_two_sows():
    table = _synthetic_table(n_sows=1, rows_per_sow=20, scale=1.0)
    with pytest.raises(ValueError, match=">=2 sows"):
        grouped_cv_train_accelerometer(table, n_splits=5, **FAST_SR_KWARGS)


def test_grouped_cv_runs_across_synthetic_sows():
    table = _synthetic_table(n_sows=4, rows_per_sow=15, scale=5.0)
    fold_results, model, scaler = grouped_cv_train_accelerometer(table, n_splits=4, **FAST_SR_KWARGS)

    assert len(fold_results) == 4
    for res in fold_results:
        assert res["n_train_sows"] + res["n_test_sows"] == 4
        assert res["n_test_sows"] >= 1
        assert res["mae_hours"] >= 0
    assert isinstance(model, OperonSymbolicRegressorWrapper)
    assert model.size > 0 and model.depth > 0


def test_evaluate_transfer_naive_vs_domain_adapted_scaling():
    """The whole point of domain-specific scaling: a model trained on one
    magnitude scale should not blow up when a fresh, target-domain scaler
    is used instead of blindly reusing the training scaler on wildly
    different raw values."""
    accelerometer_table = _synthetic_table(n_sows=3, rows_per_sow=15, scale=5.0, seed=1)
    fold_results, model, accelerometer_scaler = grouped_cv_train_accelerometer(
        accelerometer_table, n_splits=3, **FAST_SR_KWARGS
    )

    # video-scale table: same relationship, ~1e5x the accelerometer's magnitude
    video_table = _synthetic_table(n_sows=2, rows_per_sow=15, scale=5.0 * 1e5, seed=2)

    transfer = evaluate_transfer(model, video_table, accelerometer_scaler)

    assert set(transfer) == {"naive_accelerometer_scaler", "video_domain_scaler"}
    for res in transfer.values():
        assert res["n"] == len(video_table)
        assert res["mae_hours"] >= 0
    # domain-adapted MAE should be sane (same order of magnitude as the
    # target range, not blown up by a ~1e5x unit mismatch)
    assert transfer["video_domain_scaler"]["mae_hours"] < 1e4


# ---------------------------------------------------------------------------
# Per-sow baseline scaling (the third transfer strategy):
# each sow's own activity features divided by that sow's own mean activity
# in its first 48h after pen entry, rather than a population-level z-score.
# ---------------------------------------------------------------------------

def _raw_series(start: str, n_minutes: int, value) -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=n_minutes, freq="1min")
    values = value(n_minutes) if callable(value) else np.full(n_minutes, value)
    return pd.DataFrame({"timestamp": timestamps, "Acc": values})


def test_per_sow_baseline_averages_only_the_window_after_entry():
    # before entry: noise that should be excluded; in-window: a clean 10.0
    before = _raw_series("2020-01-01 00:00", 60, 1000.0)
    entry = pd.Timestamp("2020-01-01 01:00")
    window = _raw_series("2020-01-01 01:00", 48 * 60, 10.0)
    after_window = _raw_series("2020-01-03 01:00", 60, 1000.0)  # outside the 48h window too
    series = pd.concat([before, window, after_window], ignore_index=True)

    baseline = per_sow_baseline(series, entry, window_hours=48)
    assert baseline == pytest.approx(10.0)


def test_per_sow_baseline_is_nan_when_window_has_no_data():
    series = _raw_series("2020-01-05 00:00", 60, 5.0)  # entirely before entry
    entry = pd.Timestamp("2020-02-01 00:00")
    assert pd.isna(per_sow_baseline(series, entry, window_hours=48))


def test_compute_per_sow_baselines_isolates_a_failing_sow(monkeypatch):
    def fake_loader(sow):
        if sow == "bad":
            raise FileNotFoundError("no such file")
        return _raw_series("2020-01-01 00:00", 48 * 60, 10.0 if sow == "A" else 20.0)

    def fake_meta(sow):
        return pd.Series({"entry_farrowing_pen": pd.Timestamp("2020-01-01 00:00")})

    import video_activity.pooling as pooling_module
    monkeypatch.setitem(pooling_module.LOADERS, "video", fake_loader)
    monkeypatch.setattr(pooling_module, "load_meta", fake_meta)

    baselines, errors = compute_per_sow_baselines("video", ["A", "B", "bad"])

    assert baselines == {"A": pytest.approx(10.0), "B": pytest.approx(20.0)}
    assert list(errors.keys()) == ["bad"]


def test_apply_per_sow_baseline_divides_by_each_sows_own_factor():
    table = _synthetic_table(n_sows=2, rows_per_sow=5, scale=1.0, seed=3)
    table.loc[table["sow"] == "synthetic_0", ACTIVITY_FEATURES] = 10.0
    table.loc[table["sow"] == "synthetic_1", ACTIVITY_FEATURES] = 20.0
    baselines = {"synthetic_0": 5.0, "synthetic_1": 4.0}

    scaled = apply_per_sow_baseline(table, baselines)

    assert np.allclose(scaled.loc[table["sow"] == "synthetic_0", ACTIVITY_FEATURES], 2.0)
    assert np.allclose(scaled.loc[table["sow"] == "synthetic_1", ACTIVITY_FEATURES], 5.0)
    # meta features untouched
    pd.testing.assert_series_equal(scaled["crated_since"], table["crated_since"])


def test_apply_per_sow_baseline_raises_on_missing_sow():
    table = _synthetic_table(n_sows=2, rows_per_sow=5, scale=1.0, seed=4)
    with pytest.raises(ValueError, match="synthetic_1"):
        apply_per_sow_baseline(table, {"synthetic_0": 5.0})


def test_evaluate_transfer_includes_per_sow_baseline_when_given():
    accelerometer_table = _synthetic_table(n_sows=3, rows_per_sow=15, scale=5.0, seed=1)
    _, model, accelerometer_scaler = grouped_cv_train_accelerometer(
        accelerometer_table, n_splits=3, **FAST_SR_KWARGS
    )
    video_table = _synthetic_table(n_sows=2, rows_per_sow=15, scale=5.0 * 1e5, seed=2)
    baselines = {"synthetic_0": 5.0 * 1e5, "synthetic_1": 5.0 * 1e5}

    transfer = evaluate_transfer(model, video_table, accelerometer_scaler, per_sow_baselines=baselines)

    assert set(transfer) == {"naive_accelerometer_scaler", "video_domain_scaler", "per_sow_baseline"}
    assert transfer["per_sow_baseline"]["n"] == len(video_table)


def test_evaluate_transfer_per_sow_baseline_uses_only_sows_with_a_baseline():
    accelerometer_table = _synthetic_table(n_sows=3, rows_per_sow=15, scale=5.0, seed=1)
    _, model, accelerometer_scaler = grouped_cv_train_accelerometer(
        accelerometer_table, n_splits=3, **FAST_SR_KWARGS
    )
    video_table = _synthetic_table(n_sows=2, rows_per_sow=15, scale=5.0 * 1e5, seed=2)
    baselines = {"synthetic_0": 5.0 * 1e5}  # synthetic_1 has none

    transfer = evaluate_transfer(model, video_table, accelerometer_scaler, per_sow_baselines=baselines)

    n_sow0 = (video_table["sow"] == "synthetic_0").sum()
    assert transfer["per_sow_baseline"]["n"] == n_sow0


def test_evaluate_transfer_skips_per_sow_baseline_when_none_given():
    accelerometer_table = _synthetic_table(n_sows=3, rows_per_sow=15, scale=5.0, seed=1)
    _, model, accelerometer_scaler = grouped_cv_train_accelerometer(
        accelerometer_table, n_splits=3, **FAST_SR_KWARGS
    )
    video_table = _synthetic_table(n_sows=2, rows_per_sow=15, scale=5.0 * 1e5, seed=2)

    transfer = evaluate_transfer(model, video_table, accelerometer_scaler)
    assert "per_sow_baseline" not in transfer


def test_evaluate_transfer_return_predictions_matches_summary_stats():
    accelerometer_table = _synthetic_table(n_sows=3, rows_per_sow=15, scale=5.0, seed=1)
    _, model, accelerometer_scaler = grouped_cv_train_accelerometer(
        accelerometer_table, n_splits=3, **FAST_SR_KWARGS
    )
    video_table = _synthetic_table(n_sows=2, rows_per_sow=15, scale=5.0 * 1e5, seed=2)
    baselines = {"synthetic_0": 5.0 * 1e5, "synthetic_1": 5.0 * 1e5}

    results, predictions = evaluate_transfer(
        model, video_table, accelerometer_scaler, per_sow_baselines=baselines, return_predictions=True
    )

    assert set(predictions.columns) == {"strategy", "sow", "y_true_hours", "y_pred_hours"}
    assert set(predictions["strategy"].unique()) == set(results.keys())
    for strategy, res in results.items():
        sub = predictions[predictions["strategy"] == strategy]
        assert len(sub) == res["n"]
        recomputed_mae = (sub["y_true_hours"] - sub["y_pred_hours"]).abs().mean()
        assert recomputed_mae == pytest.approx(res["mae_hours"])


def test_evaluate_transfer_without_return_predictions_still_returns_plain_dict():
    accelerometer_table = _synthetic_table(n_sows=3, rows_per_sow=15, scale=5.0, seed=1)
    _, model, accelerometer_scaler = grouped_cv_train_accelerometer(
        accelerometer_table, n_splits=3, **FAST_SR_KWARGS
    )
    video_table = _synthetic_table(n_sows=2, rows_per_sow=15, scale=5.0 * 1e5, seed=2)

    result = evaluate_transfer(model, video_table, accelerometer_scaler)
    assert isinstance(result, dict)  # not a tuple — unchanged for existing callers


def test_compute_feature_sensitivity_holds_other_features_at_median():
    table = _synthetic_table(n_sows=3, rows_per_sow=15, scale=5.0, seed=1)
    model = OperonSymbolicRegressorWrapper(**FAST_SR_KWARGS)
    model.fit(table[FEATURE_COLUMNS], table["remaining_minutes"])

    sensitivity = compute_feature_sensitivity(model, table, FEATURE_COLUMNS, n_points=8)

    assert set(sensitivity["feature"].unique()) == set(FEATURE_COLUMNS)
    for feat in FEATURE_COLUMNS:
        sub = sensitivity[sensitivity["feature"] == feat]
        assert len(sub) == 8
        assert sub["reference_value"].nunique() == 1  # held constant across the sweep
        expected_median = table[feat].median()
        assert sub["reference_value"].iloc[0] == pytest.approx(expected_median)


def test_compute_feature_sensitivity_handles_constant_feature():
    table = _synthetic_table(n_sows=2, rows_per_sow=10, scale=5.0)
    table["currently_crated"] = 0  # zero-width quantile range
    model = OperonSymbolicRegressorWrapper(**FAST_SR_KWARGS)
    model.fit(table[FEATURE_COLUMNS], table["remaining_minutes"])

    sensitivity = compute_feature_sensitivity(model, table, FEATURE_COLUMNS, n_points=5)
    sub = sensitivity[sensitivity["feature"] == "currently_crated"]
    assert len(sub) == 5
    assert sub["x"].nunique() > 1  # grid still has width despite lo==hi==0


# ---------------------------------------------------------------------------
# sr_transfer.run() persists the per-row data figures.py needs, so figures
# can be redrawn later without retraining (a fresh GP run isn't reproducible).
# ---------------------------------------------------------------------------

def test_run_writes_oof_transfer_and_sensitivity_csvs_for_figures(monkeypatch, tmp_path):
    pairs = {
        "A": {"meta": True, "accelerometer": True, "video": False},
        "B": {"meta": True, "accelerometer": True, "video": False},
        "C": {"meta": True, "accelerometer": True, "video": True},
        "D": {"meta": True, "accelerometer": True, "video": True},
    }
    monkeypatch.setattr(sr_transfer.suite, "discover_pairs", lambda: pairs)
    monkeypatch.setattr(sr_transfer.suite, "complete_pairs", lambda p=None: ["C", "D"])
    monkeypatch.setattr(sr_transfer, "OUT_DIR", tmp_path)

    def fake_pool(signal, sows):
        scale = 5.0 if signal == "accelerometer" else 5.0 * 1e5
        table = _synthetic_table(n_sows=max(len(sows), 1), rows_per_sow=15, scale=scale)
        table["sow"] = table["sow"].map(dict(zip(sorted(table["sow"].unique()), sorted(sows))))
        return table, {}

    monkeypatch.setattr(sr_transfer, "build_pooled_table", fake_pool)

    def fake_baselines(signal, sows, window_hours=48):
        return {s: 5.0 * 1e5 for s in sows}, {}

    monkeypatch.setattr(sr_transfer, "compute_per_sow_baselines", fake_baselines)

    sr_transfer.run(sr_kwargs=FAST_SR_KWARGS, n_splits=2)

    assert (tmp_path / "accelerometer_cv_oof.csv").exists()
    assert (tmp_path / "transfer_predictions.csv").exists()
    assert (tmp_path / "feature_sensitivity.csv").exists()

    oof = pd.read_csv(tmp_path / "accelerometer_cv_oof.csv")
    assert set(oof.columns) == {"sow", "fold", "y_true_hours", "y_pred_hours"}

    transfer_predictions = pd.read_csv(tmp_path / "transfer_predictions.csv")
    assert set(transfer_predictions.columns) == {"strategy", "sow", "y_true_hours", "y_pred_hours"}

    sensitivity = pd.read_csv(tmp_path / "feature_sensitivity.csv")
    assert set(sensitivity["feature"].unique()) == set(FEATURE_COLUMNS)


def test_operon_wrapper_fit_predict_formula_smoke():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 3)), columns=["a", "b", "c"])
    y = 2 * X["a"] - 0.5 * X["b"] + 10

    model = OperonSymbolicRegressorWrapper(**FAST_SR_KWARGS)
    model.fit(X, y)
    pred = model.predict(X)

    assert len(pred) == len(y)
    assert model.size > 0
    assert model.depth > 0
    formula = model.formula()
    assert isinstance(formula, str) and len(formula) > 0
    assert any(name in formula for name in ["a", "b", "c"])


def test_dump_model_and_load_model_round_trip_via_formula(tmp_path):
    """pyoperon's fitted Operon::Tree can't be pickled in this build
    (its __getstate__ returns a type pybind11 fails to convert) — this
    is the actual reason dump_model/load_model serialize the formula
    string instead of the raw model object. The round trip must still
    reproduce .predict() closely."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, 3)), columns=["a", "b", "c"])
    y = 2 * X["a"] - 0.5 * X["b"] + 10 + 0.3 * X["a"] * X["c"]

    model = OperonSymbolicRegressorWrapper(population_size=200, generations=30, n_threads=1,
                                            random_state=0, optimizer_iterations=5)
    model.fit(X, y)
    pred_direct = model.predict(X)

    path = tmp_path / "model.json"
    model.dump_model(str(path))
    assert path.exists()

    from farrowing_prediction.training.operon_sr import FormulaModel
    loaded = OperonSymbolicRegressorWrapper.load_model(str(path))
    assert isinstance(loaded, FormulaModel)
    assert loaded.feature_names == ["a", "b", "c"]
    assert loaded.size == model.size
    assert loaded.depth == model.depth

    pred_loaded = loaded.predict(X)
    assert np.allclose(pred_loaded, pred_direct, atol=1e-4)


def test_evaluate_formula_matches_eval_for_simple_expression():
    from farrowing_prediction.training.operon_sr import evaluate_formula
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    result = evaluate_formula("((2.0 * x) + sqrt(y))", X)
    expected = 2.0 * X["x"].to_numpy() + np.sqrt(X["y"].to_numpy())
    assert np.allclose(result, expected)


def test_formula_model_predict_on_array_input_uses_feature_names_order():
    from farrowing_prediction.training.operon_sr import FormulaModel
    model = FormulaModel("((x) + (2.0 * y))", feature_names=["x", "y"])
    pred = model.predict(np.array([[1.0, 10.0], [2.0, 20.0]]))
    assert np.allclose(pred, [1.0 + 20.0, 2.0 + 40.0])


@requires_sample_data
def test_build_pooled_table_matches_single_sow_instance_table():
    table, errors = build_pooled_table("accelerometer", [SOW])
    assert errors == {}
    assert not table.empty
    assert (table["sow"] == SOW).all()
    assert set(FEATURE_COLUMNS).issubset(table.columns)


@requires_sample_data
def test_accelerometer_sows_discovers_sow_regardless_of_video():
    sows = accelerometer_sows()
    assert SOW in sows


# ---------------------------------------------------------------------------
# Keeping the training pool and the transfer-evaluation set disjoint: without
# this, grouped CV's "best fold" model is trained on most of the very sows
# the video-transfer check then runs on (just via accelerometer instead of
# video), so "generalizes to video" would really be "generalizes to video
# for animals it was partly trained on".
# ---------------------------------------------------------------------------

def test_accelerometer_sows_excludes_given_set():
    pairs = {
        "A": {"meta": True, "accelerometer": True, "video": False},
        "B": {"meta": True, "accelerometer": True, "video": True},
        "C": {"meta": True, "accelerometer": True, "video": True},
    }
    assert accelerometer_sows(pairs) == ["A", "B", "C"]
    assert accelerometer_sows(pairs, exclude={"B", "C"}) == ["A"]


def test_run_excludes_video_sows_from_training_by_default(monkeypatch, tmp_path):
    pairs = {
        "A": {"meta": True, "accelerometer": True, "video": False},
        "B": {"meta": True, "accelerometer": True, "video": False},
        "C": {"meta": True, "accelerometer": True, "video": True},
        "D": {"meta": True, "accelerometer": True, "video": True},
    }
    monkeypatch.setattr(sr_transfer.suite, "discover_pairs", lambda: pairs)
    monkeypatch.setattr(sr_transfer.suite, "complete_pairs", lambda p=None: ["C", "D"])
    monkeypatch.setattr(sr_transfer, "OUT_DIR", tmp_path)

    pool_calls = []

    def fake_pool(signal, sows):
        pool_calls.append((signal, sorted(sows)))
        scale = 5.0 if signal == "accelerometer" else 5.0 * 1e5
        return _synthetic_table(n_sows=max(len(sows), 1), rows_per_sow=15, scale=scale), {}

    monkeypatch.setattr(sr_transfer, "build_pooled_table", fake_pool)

    summary = sr_transfer.run(sr_kwargs=FAST_SR_KWARGS, n_splits=2)

    acc_call = next(c for c in pool_calls if c[0] == "accelerometer")
    assert acc_call[1] == ["A", "B"]  # C, D (video-paired) excluded from training
    video_call = next(c for c in pool_calls if c[0] == "video")
    assert video_call[1] == ["C", "D"]
    assert summary["held_out_transfer_eval"] is True


def test_run_falls_back_gracefully_when_every_sow_has_video(monkeypatch, tmp_path):
    """If excluding video-paired sows would leave nothing to train on, fall
    back to training on everything rather than crashing — but say clearly
    that the transfer numbers are then not a clean held-out estimate.
    """
    pairs = {
        "C": {"meta": True, "accelerometer": True, "video": True},
        "D": {"meta": True, "accelerometer": True, "video": True},
    }
    monkeypatch.setattr(sr_transfer.suite, "discover_pairs", lambda: pairs)
    monkeypatch.setattr(sr_transfer.suite, "complete_pairs", lambda p=None: ["C", "D"])
    monkeypatch.setattr(sr_transfer, "OUT_DIR", tmp_path)

    pool_calls = []

    def fake_pool(signal, sows):
        pool_calls.append((signal, sorted(sows)))
        scale = 5.0 if signal == "accelerometer" else 5.0 * 1e5
        return _synthetic_table(n_sows=max(len(sows), 1), rows_per_sow=15, scale=scale), {}

    monkeypatch.setattr(sr_transfer, "build_pooled_table", fake_pool)

    summary = sr_transfer.run(sr_kwargs=FAST_SR_KWARGS, n_splits=2)

    acc_call = next(c for c in pool_calls if c[0] == "accelerometer")
    assert acc_call[1] == ["C", "D"]  # fell back to training on everything
    assert summary["held_out_transfer_eval"] is False


def test_run_respects_include_video_sows_in_training_opt_out(monkeypatch, tmp_path):
    pairs = {
        "A": {"meta": True, "accelerometer": True, "video": False},
        "C": {"meta": True, "accelerometer": True, "video": True},
    }
    monkeypatch.setattr(sr_transfer.suite, "discover_pairs", lambda: pairs)
    monkeypatch.setattr(sr_transfer.suite, "complete_pairs", lambda p=None: ["C"])
    monkeypatch.setattr(sr_transfer, "OUT_DIR", tmp_path)

    pool_calls = []

    def fake_pool(signal, sows):
        pool_calls.append((signal, sorted(sows)))
        scale = 5.0 if signal == "accelerometer" else 5.0 * 1e5
        return _synthetic_table(n_sows=max(len(sows), 1), rows_per_sow=15, scale=scale), {}

    monkeypatch.setattr(sr_transfer, "build_pooled_table", fake_pool)

    summary = sr_transfer.run(sr_kwargs=FAST_SR_KWARGS, n_splits=2,
                               exclude_video_sows_from_training=False)

    acc_call = next(c for c in pool_calls if c[0] == "accelerometer")
    assert acc_call[1] == ["A", "C"]  # C kept in training pool on request
    assert summary["held_out_transfer_eval"] is False


def test_run_wires_in_per_sow_baseline_transfer(monkeypatch, tmp_path):
    pairs = {
        "A": {"meta": True, "accelerometer": True, "video": False},
        "B": {"meta": True, "accelerometer": True, "video": False},
        "C": {"meta": True, "accelerometer": True, "video": True},
        "D": {"meta": True, "accelerometer": True, "video": True},
    }
    monkeypatch.setattr(sr_transfer.suite, "discover_pairs", lambda: pairs)
    monkeypatch.setattr(sr_transfer.suite, "complete_pairs", lambda p=None: ["C", "D"])
    monkeypatch.setattr(sr_transfer, "OUT_DIR", tmp_path)

    def fake_pool(signal, sows):
        scale = 5.0 if signal == "accelerometer" else 5.0 * 1e5
        table = _synthetic_table(n_sows=max(len(sows), 1), rows_per_sow=15, scale=scale)
        # make the synthetic sow names line up with the real pair ids so
        # compute_per_sow_baselines (keyed by sow) can be exercised below
        table["sow"] = table["sow"].map(dict(zip(sorted(table["sow"].unique()), sorted(sows))))
        return table, {}

    monkeypatch.setattr(sr_transfer, "build_pooled_table", fake_pool)

    def fake_baselines(signal, sows, window_hours=48):
        assert signal == "video"
        # "D" has no usable baseline, mirroring a sow with no data in the window
        return {s: 5.0 * 1e5 for s in sows if s != "D"}, ({"D": "no data in window"} if "D" in sows else {})

    monkeypatch.setattr(sr_transfer, "compute_per_sow_baselines", fake_baselines)

    summary = sr_transfer.run(sr_kwargs=FAST_SR_KWARGS, n_splits=2)

    assert "per_sow_baseline" in summary["transfer"]
    assert summary["per_sow_baseline_errors"] == {"D": "no data in window"}
    assert summary["per_sow_baseline_window_hours"] == 48
    n_total = summary["n_video_instances"]
    n_c_only = summary["transfer"]["per_sow_baseline"]["n"]
    assert 0 < n_c_only < n_total  # only C's rows went into the per-sow-baseline result
