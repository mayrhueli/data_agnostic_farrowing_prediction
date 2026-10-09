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

"""Train an Operon symbolic-regression model on accelerometer data
(pooled across every sow you have, grouped k-fold CV) and test whether
it generalizes to video-derived activity data, with and without
domain-specific feature rescaling.

Usage:

    python -m video_activity.sr_transfer

With only one sow available, grouped cross-sow CV isn't meaningful (needs
>=2 sows/groups); this falls back to a single-sow train/test smoke test
so the mechanism (pooling, Operon fitting, domain scaling, transfer
evaluation) is still exercised end to end, and says so clearly in the
output — rerun with more sows under data/raw/{accelerometer,video}/ and
data/raw/meta/ for a real result.

Besides ``sr_transfer_summary.json``, this writes the per-row data
``video_activity.figures`` needs for its figures 3-5
(``accelerometer_cv_oof.csv``, ``transfer_predictions.csv``,
``feature_sensitivity.csv``) so that module can redraw them later
without retraining — important since a fresh GP run can land on a
different model (and, for the video-transfer figure, a different naive
result that overflows by a different amount), so a figure regenerated
from a *new* run wouldn't necessarily match the numbers already written
to ``sr_transfer_summary.json``. For the same reason, it also persists
the trained model itself (``model.json`` — its formula string, see
``operon_sr.OperonSymbolicRegressorWrapper.dump_model``), the
accelerometer scaler it was fit with (``accelerometer_scaler.joblib``),
and the video per-sow baselines (``per_sow_baselines.json``) —
``video_activity.online_eval`` loads these to test the *actual selected
model* in the online setting, rather than retraining a new one.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from joblib import dump

from video_activity import suite
from video_activity.pooling import build_pooled_table, compute_per_sow_baselines
from farrowing_prediction.training.instance_tables import FEATURE_COLUMNS
from farrowing_prediction.training.cross_signal import (
    grouped_cv_train_accelerometer, evaluate_transfer, fit_activity_scaler, apply_scaler,
    apply_per_sow_baseline, compute_feature_sensitivity, interleaved_split,
)
from farrowing_prediction.training.operon_sr import OperonSymbolicRegressorWrapper

OUT_DIR = suite.case_study.OUT_DIR / "sr_transfer"


def accelerometer_sows(pairs: dict = None, exclude: set = None) -> list:
    """Every sow with a metadata row and an accelerometer file, regardless
    of whether it also has video, minus ``exclude``.

    ``run()`` passes the video-paired sows as ``exclude`` by default, so
    the accelerometer training pool and the video evaluation set are
    disjoint: without this, grouped CV picks a "best fold" model trained
    on ~(n_splits-1)/n_splits of *all* accelerometer sows, which — since
    accelerometer coverage is a superset of video coverage — almost
    certainly includes most of the very animals the video-transfer check
    then runs on (just via their accelerometer trace instead of video).
    That model would have already seen most of those animals during
    training, so "generalizes to video" would really be "generalizes to
    video for animals it was partly trained on" — a much weaker and
    easily-overstated claim than a genuinely held-out evaluation set.
    """
    pairs = pairs if pairs is not None else suite.discover_pairs()
    exclude = exclude or set()
    return sorted(sow for sow, flags in pairs.items()
                  if flags["meta"] and flags["accelerometer"] and sow not in exclude)


def plot_transfer(video_table: pd.DataFrame, model: OperonSymbolicRegressorWrapper,
                   accelerometer_scaler, out_path: Path, per_sow_baselines: dict = None):
    panels = [
        ("naive (accelerometer scaler)", lambda t: apply_scaler(t, accelerometer_scaler), video_table),
        ("domain-adapted (video scaler)", lambda t: apply_scaler(t, fit_activity_scaler(t)), video_table),
    ]
    if per_sow_baselines:
        usable = video_table[video_table["sow"].isin(per_sow_baselines)]
        if not usable.empty:
            panels.append(("per-sow baseline", lambda t: apply_per_sow_baseline(t, per_sow_baselines), usable))

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.5))
    if len(panels) == 1:
        axes = [axes]
    for ax, (label, scale_fn, table) in zip(axes, panels):
        scaled = scale_fn(table)
        pred = model.predict(scaled[FEATURE_COLUMNS]) / 60
        y_true = scaled["remaining_minutes"].to_numpy() / 60
        ax.scatter(y_true, pred, s=14, alpha=0.6)
        lim = max(y_true.max(), pred.max(), 1) * 1.05
        ax.plot([0, lim], [0, lim], color="tab:blue", linewidth=1)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("Real time to farrowing [h]")
        ax.set_ylabel("Predicted time to farrowing [h]")
        ax.set_title(label)
    fig.suptitle("Accelerometer-trained Operon SR model applied to video data")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(sr_kwargs: dict = None, n_splits: int = 5, exclude_video_sows_from_training: bool = True,
        baseline_window_hours: float = 48):
    sr_kwargs = sr_kwargs or {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pairs = suite.discover_pairs()
    video_sows = suite.complete_pairs(pairs)  # need meta+accelerometer+video for the transfer check
    all_acc_sows = accelerometer_sows(pairs)
    held_out_transfer_eval = exclude_video_sows_from_training

    acc_sows = accelerometer_sows(pairs, exclude=set(video_sows)) if exclude_video_sows_from_training else all_acc_sows
    if exclude_video_sows_from_training and not acc_sows:
        print("Every accelerometer sow also has video — excluding the video-paired sows would leave "
              "0 sows to train on. Falling back to training on all accelerometer sows (transfer "
              "evaluation is then NOT a clean held-out set; see accelerometer_sows docstring).")
        acc_sows = all_acc_sows
        held_out_transfer_eval = False

    print(f"Accelerometer training pool: {len(acc_sows)} sow(s): {acc_sows}")
    print(f"Video transfer set: {len(video_sows)} sow(s): {video_sows}")
    if held_out_transfer_eval:
        overlap = set(video_sows) - set(acc_sows)
        print(f"Excluded {len(overlap)} video-paired sow(s) from the accelerometer training pool "
              f"to keep training and transfer-evaluation disjoint (see accelerometer_sows docstring). "
              f"Pass exclude_video_sows_from_training=False / --include-video-sows-in-training to disable.")

    acc_table, acc_errors = build_pooled_table("accelerometer", acc_sows)
    if acc_errors:
        print(f"Skipped {len(acc_errors)} sow(s) while pooling accelerometer data: {acc_errors}")
    if acc_table.empty:
        raise RuntimeError("No accelerometer instances could be built — check data/raw/accelerometer and data/raw/meta.")

    n_groups = acc_table["sow"].nunique()
    summary = {"n_accelerometer_sows": n_groups, "n_accelerometer_instances": len(acc_table),
               "pooling_errors": acc_errors, "held_out_transfer_eval": held_out_transfer_eval}

    if n_groups >= 2:
        print(f"Running grouped {min(n_splits, n_groups)}-fold CV across {n_groups} sows ...")
        fold_results, model, scaler, oof = grouped_cv_train_accelerometer(
            acc_table, n_splits=n_splits, return_oof=True, **sr_kwargs)
        summary["mode"] = "grouped_cv"
        summary["cv_folds"] = fold_results
        oof.to_csv(OUT_DIR / "accelerometer_cv_oof.csv", index=False)
        print(f"Wrote {OUT_DIR}/accelerometer_cv_oof.csv (out-of-fold predictions, for figures.py's "
              f"figure 3 — it reads this instead of retraining)")
    else:
        print(f"Only {n_groups} sow with accelerometer data — grouped cross-sow CV needs >=2. "
              f"Falling back to a single-sow train/test smoke test (not a real generalization "
              f"estimate); rerun with more sows for grouped_cv mode.")
        from farrowing_prediction.training.cross_signal import _pcc

        train, test = interleaved_split(acc_table)
        scaler = fit_activity_scaler(train)
        train_s, test_s = apply_scaler(train, scaler), apply_scaler(test, scaler)
        model = OperonSymbolicRegressorWrapper(**sr_kwargs)
        model.fit(train_s[FEATURE_COLUMNS], train_s["remaining_minutes"])
        _, mae, _ = model.evaluate(test_s[FEATURE_COLUMNS], test_s["remaining_minutes"])
        pred = model.predict(test_s[FEATURE_COLUMNS])
        summary["mode"] = "single_sow_smoke_test"
        summary["single_sow_result"] = {
            "n_train": len(train), "n_test": len(test),
            "mae_hours": mae / 60,
            "pcc": _pcc(test_s["remaining_minutes"].to_numpy(), pred),
        }

    summary["formula"] = model.formula()
    summary["size"] = model.size
    summary["depth"] = model.depth
    summary["features"] = model.feature_names
    print("\nBest model:")
    print(f"  size={model.size}, depth={model.depth}")
    print(f"  formula: {model.formula()}")

    model.dump_model(str(OUT_DIR / "model.json"))
    dump(scaler, OUT_DIR / "accelerometer_scaler.joblib")
    print(f"Wrote {OUT_DIR}/model.json, accelerometer_scaler.joblib (for video_activity.online_eval)")

    sensitivity = compute_feature_sensitivity(model, apply_scaler(acc_table, scaler), FEATURE_COLUMNS)
    sensitivity.to_csv(OUT_DIR / "feature_sensitivity.csv", index=False)
    print(f"Wrote {OUT_DIR}/feature_sensitivity.csv (for figures.py's figure 5)")

    video_table, video_errors = build_pooled_table("video", video_sows)
    if video_errors:
        print(f"Skipped {len(video_errors)} sow(s) while pooling video data: {video_errors}")
    summary["n_video_sows"] = video_table["sow"].nunique() if not video_table.empty else 0
    summary["n_video_instances"] = len(video_table)
    summary["video_pooling_errors"] = video_errors

    if not video_table.empty:
        per_sow_baselines, baseline_errors = compute_per_sow_baselines(
            "video", video_table["sow"].unique().tolist(), window_hours=baseline_window_hours)
        if baseline_errors:
            print(f"Skipped {len(baseline_errors)} sow(s) for per-sow baseline scaling "
                  f"(no video data in their first {baseline_window_hours}h after pen entry): {baseline_errors}")
        summary["per_sow_baseline_window_hours"] = baseline_window_hours
        summary["per_sow_baseline_errors"] = baseline_errors

        with open(OUT_DIR / "per_sow_baselines.json", "w") as f:
            json.dump(per_sow_baselines, f, indent=2, default=str)

        transfer, transfer_predictions = evaluate_transfer(
            model, video_table, scaler, per_sow_baselines=per_sow_baselines, return_predictions=True)
        summary["transfer"] = transfer
        eval_kind = "held-out (model never saw these sows during training)" if held_out_transfer_eval \
            else "NOT held-out (some/all of these sows' accelerometer data was in the training pool)"
        print(f"\nTransfer to video data [{eval_kind}]:")
        for label, res in transfer.items():
            print(f"  {label}: n={res['n']}, MAE={res['mae_hours']:.2f}h, PCC={res['pcc']:.3f}")
        plot_transfer(video_table, model, scaler, OUT_DIR / "transfer_scatter.png",
                       per_sow_baselines=per_sow_baselines)
        transfer_predictions.to_csv(OUT_DIR / "transfer_predictions.csv", index=False)
        print(f"Wrote {OUT_DIR}/transfer_predictions.csv (per-row predictions per strategy, for "
              f"figures.py's figure 4 — it reads this instead of recomputing the transfer)")
    else:
        print("\nNo video-paired sows available for the transfer check.")
        summary["transfer"] = {}

    with open(OUT_DIR / "sr_transfer_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote {OUT_DIR}/sr_transfer_summary.json")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--population-size", type=int, default=1000)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-threads", type=int, default=1,
                         help="pyoperon's multi-threaded evaluator has an intermittent race in this build "
                              "(see operon_sr.py); defaults to 1 for reliability, retries automatically if >1.")
    parser.add_argument("--max-length", type=int, default=25, help="max expression tree size (nodes)")
    parser.add_argument("--max-depth", type=int, default=7, help="max expression tree depth")
    parser.add_argument("--optimizer-iterations", type=int, default=10,
                         help="Levenberg-Marquardt local search iterations per individual per generation. "
                              "pyoperon's own default is 0 (off) — see operon_sr.py for why that's a trap.")
    parser.add_argument("--single-objective", action="store_true",
                         help="Optimize r2 only (pyoperon's own default), instead of the default here "
                              "of ['r2', 'length'] (NSGA2, active parsimony pressure toward smaller trees).")
    parser.add_argument("--allowed-symbols", type=str, default="add,sub,mul,aq,constant,variable",
                         help="comma-separated Operon symbols. Default swaps div for aq (analytic quotient) "
                              "to avoid the near-zero-denominator blowups plain division is prone to.")
    parser.add_argument("--include-video-sows-in-training", action="store_true",
                         help="By default, sows that have video are excluded from the accelerometer "
                              "training pool so the video-transfer evaluation is on genuinely unseen "
                              "animals, not just an unseen signal for animals the model was partly "
                              "trained on. Pass this to include them (e.g. if excluding them would "
                              "leave too little training data) — the transfer numbers are then not a "
                              "clean held-out estimate, and the output says so.")
    parser.add_argument("--baseline-window-hours", type=float, default=48,
                         help="Window after pen entry used for the per-sow baseline transfer strategy "
                              "(each sow's activity features divided by its own mean activity in this "
                              "window). Default 48h.")
    args = parser.parse_args()

    run(
        sr_kwargs={
            "population_size": args.population_size,
            "generations": args.generations,
            "n_threads": args.n_threads,
            "max_length": args.max_length,
            "max_depth": args.max_depth,
            "optimizer_iterations": args.optimizer_iterations,
            "objectives": ["r2"] if args.single_objective else ["r2", "length"],
            "allowed_symbols": args.allowed_symbols,
        },
        n_splits=args.n_splits,
        exclude_video_sows_from_training=not args.include_video_sows_in_training,
        baseline_window_hours=args.baseline_window_hours,
    )
