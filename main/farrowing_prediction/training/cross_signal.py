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

"""Train a symbolic regression model on accelerometer data (pooled
across sows, grouped 5-fold CV) and test whether it generalizes to
video-derived activity data.

The two signals live on wildly different scales (accelerometer magnitude
is O(1-10) m/s^2; video's Flow_totals is O(1e5-1e6)), so a model fit on
raw accelerometer feature values has no reason to produce sane output on
raw video feature values. The activity-magnitude features (mean_*,
trend_*) are therefore standardized *per signal domain* — a scaler fit on
each domain's own feature statistics, never on labels, so it's legitimate
to fit one on the video table at evaluation time without any video
ground truth. The two crating features (currently_crated, crated_since)
describe the same real-world event regardless of which sensor recorded
it, so they are never rescaled.

``evaluate_transfer`` reports the naive case (reusing the
accelerometer-fitted scaler on raw video features), the domain-adapted
case (a fresh, video-only, unsupervised scaler), and — when per-sow
baselines are supplied — the per-sow case (each sow's own activity
features divided by that sow's mean activity in its first 48h after pen
entry, a ratio rather than a z-score). Unlike the domain-level scaler,
which needs the whole video population's statistics (including the test
sow's own instances) and so is only available in hindsight, the per-sow
baseline needs just that one animal's own early data and is therefore
usable online for a brand-new sow.
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from farrowing_prediction.training.instance_tables import FEATURE_COLUMNS
from farrowing_prediction.training.operon_sr import OperonSymbolicRegressorWrapper

ACTIVITY_FEATURES = FEATURE_COLUMNS[:8]   # mean_*/trend_* — signal-magnitude-derived, rescaled per domain
META_FEATURES = FEATURE_COLUMNS[8:]       # currently_crated, crated_since — signal-independent, never rescaled

assert ACTIVITY_FEATURES == ["mean_day", "mean_night", "mean_24", "mean_48",
                              "mean_last_12", "mean_last_24", "trend_last_12", "trend_last_24"]
assert META_FEATURES == ["currently_crated", "crated_since"]


def fit_activity_scaler(table: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on a table's own activity features only —
    unsupervised, so this can legitimately be fit on the video table at
    evaluation time (no video labels involved)."""
    return StandardScaler().fit(table[ACTIVITY_FEATURES])


def apply_scaler(table: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    scaled = table.copy()
    scaled[ACTIVITY_FEATURES] = scaler.transform(table[ACTIVITY_FEATURES])
    return scaled


def apply_per_sow_baseline(table: pd.DataFrame, baselines: dict) -> pd.DataFrame:
    """Divide each row's activity features by that row's sow's own
    baseline (see ``video_activity.pooling.compute_per_sow_baselines``) —
    a ratio to the animal's own resting level, not a z-score. Every sow
    present in ``table`` must have an entry in ``baselines``; the caller
    (``evaluate_transfer``) restricts ``table`` to sows with a usable
    baseline before calling this.
    """
    if "sow" not in table.columns:
        raise ValueError("table must have a 'sow' column (see pooling.build_pooled_table)")
    missing = set(table["sow"]) - set(baselines)
    if missing:
        raise ValueError(f"no per-sow baseline available for sow(s): {sorted(missing)}")

    scaled = table.copy()
    factors = scaled["sow"].map(baselines)
    scaled[ACTIVITY_FEATURES] = scaled[ACTIVITY_FEATURES].div(factors, axis=0)
    return scaled


def _pcc(y_true, y_pred):
    if len(y_true) < 2 or np.std(y_pred) == 0:
        return float("nan")
    return pearsonr(y_true, y_pred)[0]


def interleaved_split(table: pd.DataFrame, train_fraction: float = 0.7):
    """Split by interleaving every k-th row into the test set, rather than
    a chronological block split. With a single sow, ``remaining_minutes``
    falls monotonically over time, so a block split forces the model to
    extrapolate (train on "far from farrowing", test on "close to
    farrowing") instead of testing in-range interpolation. Interleaving
    keeps both sets spanning the full pre-farrowing range, which is the
    fairest in-sample check available without a second sow to hold out.
    Used as a fallback when too few sows are available for grouped CV.
    """
    k = max(2, round(1 / (1 - train_fraction)))
    test_mask = (np.arange(len(table)) % k) == 0
    test = table.loc[test_mask].reset_index(drop=True)
    train = table.loc[~test_mask].reset_index(drop=True)
    return train, test


def grouped_cv_train_accelerometer(table: pd.DataFrame, n_splits: int = 5,
                                    return_oof: bool = False, **sr_kwargs):
    """Grouped k-fold CV, grouped by sow, on the pooled accelerometer
    table. Returns (fold_results, best_model, best_scaler)
    — the model/scaler from the fold with the lowest held-out MAE, kept
    for the video-transfer check.

    With ``return_oof=True``, also returns a fourth item: a DataFrame of
    every fold's held-out predictions pooled together (``sow``, ``fold``,
    ``y_true_hours``, ``y_pred_hours``) — out-of-fold, so each row's
    prediction came from a model that never saw that sow during training.
    Opt-in (not part of ``fold_results``, which stays summary-only) so
    existing 3-item unpacking keeps working; it's for plotting the
    accelerometer-side cross-validation performance (true vs. predicted
    time-to-farrowing across every sow, not just fold-level MAE/PCC).
    """
    if "sow" not in table.columns:
        raise ValueError("table must have a 'sow' column (see pooling.build_pooled_table)")

    groups = table["sow"]
    n_groups = groups.nunique()
    if n_groups < 2:
        raise ValueError(
            f"Grouped cross-validation needs >=2 sows, got {n_groups}. With a single "
            f"sow, use interleaved_split instead — this function is for pooled, "
            f"multi-sow training."
        )
    n_splits = min(n_splits, n_groups)
    kf = GroupKFold(n_splits=n_splits)

    fold_results = []
    oof_rows = []
    best = None
    for fold, (train_idx, test_idx) in enumerate(kf.split(table, groups=groups)):
        train, test = table.iloc[train_idx], table.iloc[test_idx]

        scaler = fit_activity_scaler(train)
        train_s, test_s = apply_scaler(train, scaler), apply_scaler(test, scaler)

        model = OperonSymbolicRegressorWrapper(**sr_kwargs)
        model.fit(train_s[FEATURE_COLUMNS], train_s["remaining_minutes"])
        _, mae, _ = model.evaluate(test_s[FEATURE_COLUMNS], test_s["remaining_minutes"])
        pred = model.predict(test_s[FEATURE_COLUMNS])

        fold_results.append({
            "fold": fold,
            "n_train": len(train), "n_test": len(test),
            "n_train_sows": train["sow"].nunique(), "n_test_sows": test["sow"].nunique(),
            "mae_hours": mae / 60,
            "pcc": _pcc(test_s["remaining_minutes"].to_numpy(), pred),
            "size": model.size, "depth": model.depth,
        })

        if return_oof:
            oof_rows.append(pd.DataFrame({
                "sow": test["sow"].to_numpy(),
                "fold": fold,
                "y_true_hours": test_s["remaining_minutes"].to_numpy() / 60,
                "y_pred_hours": pred / 60,
            }))

        if best is None or mae < best["mae"]:
            best = {"model": model, "scaler": scaler, "mae": mae}

    if return_oof:
        oof = pd.concat(oof_rows, ignore_index=True) if oof_rows else pd.DataFrame(
            columns=["sow", "fold", "y_true_hours", "y_pred_hours"])
        return fold_results, best["model"], best["scaler"], oof
    return fold_results, best["model"], best["scaler"]


def evaluate_transfer(model: OperonSymbolicRegressorWrapper, video_table: pd.DataFrame,
                       accelerometer_scaler: StandardScaler, per_sow_baselines: dict = None,
                       return_predictions: bool = False):
    """Evaluate an accelerometer-trained model on video data under each
    available scaling strategy: naive (reuse the accelerometer scaler),
    domain-level (a fresh video-only scaler), and — if ``per_sow_baselines``
    is given (see ``video_activity.pooling.compute_per_sow_baselines``) —
    per-sow (each sow's own 10/90... features divided by its own baseline).

    A sow missing from ``per_sow_baselines`` (e.g. no video data in its
    first 48h after pen entry) is simply left out of the per-sow result,
    not out of naive/domain-level, which don't need it; that result's own
    ``n`` reflects the subset actually used.

    With ``return_predictions=True``, also returns a long DataFrame of
    every row's prediction under every strategy (``strategy``, ``sow``,
    ``y_true_hours``, ``y_pred_hours``) — for persisting to disk so a
    plot can be redrawn later from the stored result, without needing
    the (expensive, and for a fresh GP run non-reproducible) model again.
    Opt-in, default-False, so existing dict-only callers are unaffected.
    """
    empty_predictions = pd.DataFrame(columns=["strategy", "sow", "y_true_hours", "y_pred_hours"])
    if video_table.empty:
        return ({}, empty_predictions) if return_predictions else {}

    results = {}
    pred_rows = []
    for label, scaler in [
        ("naive_accelerometer_scaler", accelerometer_scaler),
        ("video_domain_scaler", fit_activity_scaler(video_table)),
    ]:
        scaled = apply_scaler(video_table, scaler)
        pred = model.predict(scaled[FEATURE_COLUMNS])
        y_true = scaled["remaining_minutes"].to_numpy()
        results[label] = {
            "n": len(video_table),
            "mae_hours": mean_absolute_error(y_true, pred) / 60,
            "pcc": _pcc(y_true, pred),
        }
        if return_predictions:
            pred_rows.append(pd.DataFrame({
                "strategy": label, "sow": scaled["sow"].to_numpy(),
                "y_true_hours": y_true / 60, "y_pred_hours": pred / 60,
            }))

    if per_sow_baselines:
        usable = video_table[video_table["sow"].isin(per_sow_baselines)]
        if not usable.empty:
            scaled = apply_per_sow_baseline(usable, per_sow_baselines)
            pred = model.predict(scaled[FEATURE_COLUMNS])
            y_true = scaled["remaining_minutes"].to_numpy()
            results["per_sow_baseline"] = {
                "n": len(usable),
                "mae_hours": mean_absolute_error(y_true, pred) / 60,
                "pcc": _pcc(y_true, pred),
            }
            if return_predictions:
                pred_rows.append(pd.DataFrame({
                    "strategy": "per_sow_baseline", "sow": scaled["sow"].to_numpy(),
                    "y_true_hours": y_true / 60, "y_pred_hours": pred / 60,
                }))

    if return_predictions:
        predictions = pd.concat(pred_rows, ignore_index=True) if pred_rows else empty_predictions
        return results, predictions
    return results


def compute_feature_sensitivity(model: OperonSymbolicRegressorWrapper, reference_table: pd.DataFrame,
                                 features: list, n_points: int = 50) -> pd.DataFrame:
    """1D partial-dependence-style curves read directly off a fitted
    model: for each feature, vary it across its 5th-95th percentile in
    ``reference_table`` while holding every other feature at its median,
    and record the model's own (exact, not sampled) prediction. This is
    the exact-formula alternative to SHAP described in
    ``video_activity.figures.plot_feature_sensitivity`` — kept here,
    separate from plotting, so the curve data can be computed once
    (right after training, from the live model) and persisted, and the
    figure redrawn later from that CSV without needing the model again.

    Returns a long DataFrame: ``feature``, ``x``, ``y_pred_hours``,
    ``reference_value`` (that feature's held-at-median value, for
    marking it on the plot).
    """
    baseline = reference_table[features].median()
    rows = []
    for feat in features:
        lo, hi = reference_table[feat].quantile([0.05, 0.95])
        if lo == hi:
            lo, hi = reference_table[feat].min(), reference_table[feat].max()
        if lo == hi: 
            lo, hi = lo - 1, hi + 1
        grid = np.linspace(lo, hi, n_points)

        X = pd.DataFrame(np.tile(baseline.values, (n_points, 1)), columns=features)
        X[feat] = grid
        pred = model.predict(X) / 60

        rows.append(pd.DataFrame({
            "feature": feat, "x": grid, "y_pred_hours": pred, "reference_value": baseline[feat],
        }))
    return pd.concat(rows, ignore_index=True)
