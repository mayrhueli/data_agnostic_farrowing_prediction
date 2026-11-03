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

"""Tests for the styled figure set (figures.py). figures.py only reads
already-written results and plots them — it does not retrain a model or
rerun detection — so these tests feed it synthetic CSVs/DataFrames in the
exact shape sr_transfer.py writes, rather than running any pipeline.
"""
import json

import numpy as np
import pandas as pd
import pytest

from video_activity import figures


def test_palette_has_six_colors_and_is_applied():
    assert len(figures.PALETTE) == 6
    assert all(c.startswith("#") and len(c) == 7 for c in figures.PALETTE)
    figures.apply_style()
    import matplotlib as mpl
    cycle_colors = [d["color"] for d in mpl.rcParams["axes.prop_cycle"]]
    assert cycle_colors == figures.PALETTE


def test_apply_style_keeps_text_black():
    figures.apply_style()
    import matplotlib as mpl
    for key in ["text.color", "axes.labelcolor", "axes.titlecolor", "xtick.color",
                "ytick.color", "legend.labelcolor"]:
        assert mpl.rcParams[key] == "black"


def _fake_oof(n_sows=4, rows_per_sow=10, n_folds=4, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sows):
        y_true = np.linspace(48, 1, rows_per_sow)
        y_pred = y_true + rng.normal(0, 5, rows_per_sow)
        rows.append(pd.DataFrame({
            "sow": f"sow_{s}", "fold": s % n_folds,
            "y_true_hours": y_true, "y_pred_hours": y_pred,
        }))
    return pd.concat(rows, ignore_index=True)


def test_plot_accelerometer_cv_runs(tmp_path):
    out = tmp_path / "cv.png"
    figures.plot_accelerometer_cv(_fake_oof(), out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_accelerometer_cv_handles_a_single_row(tmp_path):
    oof = pd.DataFrame({"sow": ["A"], "fold": [0], "y_true_hours": [10.0], "y_pred_hours": [12.0]})
    out = tmp_path / "cv_single.png"
    figures.plot_accelerometer_cv(oof, out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_accelerometer_cv_bins_by_true_time_to_farrowing():
    """The whole point of the replacement figure: error grouped by how
    far out the true time-to-farrowing was, not a raw scatter cloud."""
    oof = pd.DataFrame({
        "sow": ["A"] * 4,
        "fold": [0] * 4,
        "y_true_hours": [2.0, 10.0, 14.0, 40.0],
        "y_pred_hours": [3.0, 11.0, 16.0, 41.0],  # +1h error in bin1, +2h in bin2
    })
    bins = pd.cut(oof["y_true_hours"], np.arange(0, 48 + 12, 12), right=False)
    assert list(bins.cat.codes) == [0, 0, 1, 3]  # 0-12h, 0-12h, 12-24h, 36-48h


def _fake_transfer_predictions(strategies, n=20, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for strategy in strategies:
        y_true = np.linspace(48, 1, n)
        y_pred = y_true + rng.normal(0, 8, n)
        rows.append(pd.DataFrame({
            "strategy": strategy, "sow": [f"sow_{i % 3}" for i in range(n)],
            "y_true_hours": y_true, "y_pred_hours": y_pred,
        }))
    return pd.concat(rows, ignore_index=True)


def test_plot_transfer_styled_runs_with_two_and_three_strategies(tmp_path):
    out2 = tmp_path / "transfer_2panel.png"
    figures.plot_transfer_styled(
        _fake_transfer_predictions(["naive_accelerometer_scaler", "video_domain_scaler"]), out2)
    assert out2.exists() and out2.stat().st_size > 0

    out3 = tmp_path / "transfer_3panel.png"
    figures.plot_transfer_styled(
        _fake_transfer_predictions(
            ["naive_accelerometer_scaler", "video_domain_scaler", "per_sow_baseline"]), out3)
    assert out3.exists() and out3.stat().st_size > 0


def _fake_sensitivity(features, n_points=10, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for feat in features:
        x = np.linspace(-1, 1, n_points)
        rows.append(pd.DataFrame({
            "feature": feat, "x": x, "y_pred_hours": rng.normal(20, 2, n_points),
            "reference_value": 0.0,
        }))
    return pd.concat(rows, ignore_index=True)


def test_plot_feature_sensitivity_runs_for_every_feature(tmp_path):
    from farrowing_prediction.training.instance_tables import FEATURE_COLUMNS
    out = tmp_path / "sensitivity.png"
    figures.plot_feature_sensitivity(_fake_sensitivity(FEATURE_COLUMNS), out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_nest_building_comparison_requires_existing_summary(tmp_path, monkeypatch):
    """Figure 2 must not fall back to rerunning detection (BEAST is
    stochastic) — a missing <sow>_summary.json should fail loudly and
    name the command that produces it, not silently recompute."""
    with pytest.raises(FileNotFoundError, match="case_study"):
        figures.plot_nest_building_comparison(["nonexistent_sow"], tmp_path / "fig2.png",
                                                results_dir=tmp_path)


def test_plot_nest_building_comparison_reads_existing_summary(tmp_path, monkeypatch):
    sow = "sample_sow"
    from video_activity.ingestion import ACCELEROMETER_DIR, VIDEO_DIR, META_DIR
    requires_sample_data = (
        (ACCELEROMETER_DIR / f"{sow}.csv").exists()
        and (VIDEO_DIR / f"{sow}_activity.csv").exists()
        and (META_DIR / f"{sow}.csv").exists()
    )
    if not requires_sample_data:
        pytest.skip("sample sow data not present under data/raw")

    from video_activity import case_study
    summary = case_study.main(sow)
    (tmp_path / f"{sow}_summary.json").write_text(json.dumps(summary, default=str))

    out = tmp_path / "fig2.png"
    figures.plot_nest_building_comparison([sow], out, results_dir=tmp_path)
    assert out.exists() and out.stat().st_size > 0


def test_generate_all_skips_missing_result_files_without_crashing(tmp_path, monkeypatch):
    """With no sr_transfer.py/named_sow_eval.py output present yet,
    figures 3-7 should be skipped with a message, not crash the whole
    run (figures 1-2 still need real sow data, so this only checks the
    skip path doesn't raise for figures 3-7)."""
    from video_activity.ingestion import ACCELEROMETER_DIR, VIDEO_DIR, META_DIR
    sow = "sample_sow"
    if not ((ACCELEROMETER_DIR / f"{sow}.csv").exists()
            and (VIDEO_DIR / f"{sow}_activity.csv").exists()
            and (META_DIR / f"{sow}.csv").exists()):
        pytest.skip("sample sow data not present under data/raw")

    from video_activity import case_study
    case_study.main(sow)  # ensures <sow>_summary.json exists for figure 2

    monkeypatch.setattr(figures, "OUT_DIR", tmp_path / "figures")
    figures.generate_all([sow], results_dir=tmp_path / "nonexistent_results_dir",
                          named_sow_eval_dir=tmp_path / "nonexistent_named_sow_eval_dir")

    assert (tmp_path / "figures" / "1_signal_comparison.png").exists()
    assert (tmp_path / "figures" / "2_nest_building_comparison.png").exists()
    assert not (tmp_path / "figures" / "3_accelerometer_cv.png").exists()
    assert not (tmp_path / "figures" / "4_transfer.png").exists()
    assert not (tmp_path / "figures" / "5_feature_sensitivity.png").exists()
    assert not (tmp_path / "figures" / "6_anchored_prediction.png").exists()
    assert not (tmp_path / "figures" / "7_online_prediction.png").exists()


# ---------------------------------------------------------------------------
# Figures 6-7: per-named-animal prediction scatter (anchored / online)
# ---------------------------------------------------------------------------

def _fake_anchored_predictions(sows, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for sow in sows:
        y_true = np.linspace(48, 1, 20)
        for signal, strategies in [
            ("accelerometer", ["native_accelerometer_scaler"]),
            ("video", ["naive_accelerometer_scaler", "video_domain_scaler", "per_sow_baseline"]),
        ]:
            for strategy in strategies:
                scale = 1e5 if strategy == "naive_accelerometer_scaler" else 1.0
                y_pred = y_true * scale + rng.normal(0, 2, 20)
                rows.append(pd.DataFrame({"sow": sow, "signal": signal, "scaling_strategy": strategy,
                                           "y_true_hours": y_true, "y_pred_hours": y_pred}))
    return pd.concat(rows, ignore_index=True)


def test_plot_anchored_prediction_scatter_runs_for_multiple_sows(tmp_path):
    sows = ["147002 7", "147037 18", "147145 2"]
    out = tmp_path / "fig6.png"
    figures.plot_anchored_prediction_scatter(sows, _fake_anchored_predictions(sows), out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_anchored_prediction_scatter_handles_missing_signal_for_one_sow(tmp_path):
    sows = ["A", "B"]
    predictions = _fake_anchored_predictions(["A"])  # B has no data at all
    out = tmp_path / "fig6_partial.png"
    figures.plot_anchored_prediction_scatter(sows, predictions, out)
    assert out.exists() and out.stat().st_size > 0


def _fake_online_predictions(sows, methods=("NestDetect",), seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for sow in sows:
        for method in methods:
            y_true = np.linspace(24, 1, 10)
            for signal, strategies in [
                ("accelerometer", ["native_accelerometer_scaler"]),
                ("video", ["naive_accelerometer_scaler", "video_domain_scaler", "per_sow_baseline"]),
            ]:
                for strategy in strategies:
                    scale = 1e5 if strategy == "naive_accelerometer_scaler" else 1.0
                    y_pred = y_true * scale + rng.normal(0, 2, 10)
                    rows.append(pd.DataFrame({
                        "sow": sow, "signal": signal, "method": method, "scaling_strategy": strategy,
                        "y_true_hours": y_true, "y_pred_hours": y_pred,
                    }))
    return pd.concat(rows, ignore_index=True)


def test_plot_online_prediction_scatter_runs_for_multiple_sows(tmp_path):
    sows = ["147002 7", "147037 18", "147145 2"]
    out = tmp_path / "fig7.png"
    figures.plot_online_prediction_scatter(sows, _fake_online_predictions(sows), out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_online_prediction_scatter_shows_fallback_when_detector_never_fired(tmp_path):
    """A sow where the chosen method produced no online rows at all
    (never fired before farrowing) must render a fallback panel, not
    crash on an empty group."""
    predictions = _fake_online_predictions(["A"], methods=("NestDetect",))
    out = tmp_path / "fig7_missing_method.png"
    # ask for CUSUM, which has no rows in the fixture -> every panel should fall back
    figures.plot_online_prediction_scatter(["A"], predictions, out, method="CUSUM")
    assert out.exists() and out.stat().st_size > 0
