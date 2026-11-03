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

"""The seven figure types:

1. ``plot_signal_comparison``          -- accelerometer vs. video activity, per sow
2. ``plot_nest_building_comparison``   -- detection changepoints over activity, per sow
3. ``plot_accelerometer_cv``           -- out-of-fold prediction error vs. true time-to-
                                           farrowing, binned (see its docstring for why this
                                           replaced a plain true-vs-predicted scatter)
4. ``plot_transfer_styled``            -- the video-transfer scatter (naive / domain / per-sow)
5. ``plot_feature_sensitivity``        -- see its docstring for why this replaces SHAP here
6. ``plot_anchored_prediction_scatter`` -- true vs. predicted, farrowing-anchored, for named sows
7. ``plot_online_prediction_scatter``   -- true vs. predicted, online setting, for named sows

This module only reads and plots; it never retrains or reruns detection.
Figures 3-5 read their data from the CSVs ``video_activity.sr_transfer``
writes alongside its JSON summary (``accelerometer_cv_oof.csv``,
``transfer_predictions.csv``, ``feature_sensitivity.csv``); figures 6-7
read ``video_activity.named_sow_eval``'s output
(``anchored_predictions.csv``, ``online_predictions.csv``) — run both
first:

    python -m video_activity.sr_transfer
    python -m video_activity.named_sow_eval --sow "sample_sow"
    python -m video_activity.figures --sows "sample_sow"

Figure 2 reads each sow's detection result from case_study.py's already
-written ``<sow>_summary.json`` (``python -m video_activity.case_study``
or ``suite.py``) rather than rerunning CUSUM/BEAST/NestDetect — BEAST in
particular is stochastic (MCMC), so rerunning it here would draw a
*different* changepoint than whatever was already reported, silently
disagreeing with your own written results. Figure 1 is the only one that
reads raw sensor files directly — there is no "result" to reuse there,
it's just the input data (deterministic, not a model output).

Missing an input file fails with a clear message naming the command that
produces it, rather than silently falling back to recomputing it.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from video_activity import suite
from video_activity.ingestion import load_accelerometer_series, load_video_series, load_meta
from video_activity.case_study import crop_series

OUT_DIR = suite.case_study.OUT_DIR / "figures"
SR_TRANSFER_DIR = suite.case_study.OUT_DIR / "sr_transfer"
NAMED_SOW_EVAL_DIR = suite.case_study.OUT_DIR / "named_sow_eval"

# Tableau-style qualitative palette, as given: blue, orange, green, red,
# purple, teal. Used in this fixed order everywhere so a color means the
# same thing across every figure in the set (e.g. palette[0] is always
# "accelerometer" in figures 1-2).
PALETTE = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1", "#76B7B2"]

VIDEO_STRATEGIES_FOR_SCATTER = [
    ("video_domain_scaler", "domain-adapted", PALETTE[0]),
    ("per_sow_baseline", "per-sow baseline", PALETTE[2]),
]


def apply_style():
    mpl.rcParams.update({
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "text.color": "black",
        "axes.labelcolor": "black",
        "axes.titlecolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "legend.labelcolor": "black",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "legend.frameon": False,
    })


# ---------------------------------------------------------------------------
# 1. Accelerometer vs. video activity, per sow (raw input data, not a "result")
# ---------------------------------------------------------------------------

def plot_signal_comparison(sows: list, out_path: Path):
    apply_style()
    fig, axes = plt.subplots(len(sows), 1, figsize=(11, 3.2 * len(sows)), squeeze=False)

    for row, sow in enumerate(sows):
        ax_acc = axes[row, 0]
        meta = load_meta(sow)
        farrowing_time = meta["start_farrowing"]
        acc = load_accelerometer_series(sow)
        video = load_video_series(sow)

        acc_roll = acc["Acc"].rolling(720, min_periods=1).mean()
        video_roll = video["Acc"].rolling(720, min_periods=1).mean()
        acc_days = (acc["timestamp"] - farrowing_time).dt.total_seconds() / 86400
        video_days = (video["timestamp"] - farrowing_time).dt.total_seconds() / 86400

        ax_acc.plot(acc_days, acc_roll, color=PALETTE[0], linewidth=1.2,
                    label="accelerometer (12h rolling mean)")
        ax_acc.set_ylabel("Accelerometer magnitude")

        ax_video = ax_acc.twinx()
        ax_video.plot(video_days, video_roll, color=PALETTE[1], linewidth=1.2,
                       label="video optical flow (12h rolling mean)")
        ax_video.set_ylabel("Video Flow_totals")
        ax_video.grid(False)

        ax_acc.axvline(0, color="black", linewidth=1.2, linestyle="--")
        ax_acc.set_title(f"Sow {sow}", fontsize=10)

        lines1, labels1 = ax_acc.get_legend_handles_labels()
        lines2, labels2 = ax_video.get_legend_handles_labels()
        ax_acc.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    axes[-1, 0].set_xlabel("Days relative to onset of farrowing")
    fig.suptitle("Accelerometer vs. video-derived activity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Nest-building detection, per sow — reads case_study.py's <sow>_summary.json
# ---------------------------------------------------------------------------

def plot_nest_building_comparison(sows: list, out_path: Path, results_dir: Path = None):
    apply_style()
    results_dir = results_dir or suite.case_study.OUT_DIR
    method_colors = {"NestDetect": PALETTE[3], "CUSUM": PALETTE[2], "BEAST": PALETTE[4]}
    fig, axes = plt.subplots(len(sows), 2, figsize=(12, 3.4 * len(sows)), squeeze=False)

    for row, sow in enumerate(sows):
        summary_path = results_dir / f"{sow}_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"{summary_path} not found. Figure 2 reads each sow's already-computed detection "
                f"result instead of rerunning the detectors (BEAST is stochastic, so rerunning it "
                f"here could draw a different changepoint than whatever's already been reported). "
                f"Run `python -m video_activity.case_study --sow \"{sow}\"` (or suite.py, which does "
                f"every sow at once) first."
            )
        with open(summary_path) as f:
            det_summary = json.load(f)

        farrowing_time = pd.Timestamp(det_summary["farrowing_time"])
        meta = load_meta(sow)
        entry = meta["entry_farrowing_pen"]
        detection_end = farrowing_time + pd.Timedelta(hours=12)

        acc_det = crop_series(load_accelerometer_series(sow), entry, detection_end)
        video_det = crop_series(load_video_series(sow), entry, detection_end)

        for col, (df, signal_name, display_name) in enumerate([
            (acc_det, "accelerometer", "accelerometer"),
            (video_det, "video", "video (optical flow)"),
        ]):
            ax = axes[row, col]
            rolling = df["Acc"].rolling(720, min_periods=1).mean()
            days = (df["timestamp"] - farrowing_time).dt.total_seconds() / 86400
            ax.plot(days, rolling, color=PALETTE[0], linewidth=1.1, label="rolling mean 12h")

            for method, det in det_summary["detection"][signal_name].items():
                if det is None or det["changepoint"] is None:
                    continue
                cp = pd.Timestamp(det["changepoint"])
                x = (cp - farrowing_time).total_seconds() / 86400
                ax.axvline(x, color=method_colors[method], linestyle="--", linewidth=1.3, label=method)

            ax.axvline(0, color="black", linewidth=1.3, label="farrowing onset")
            ax.set_title(f"Sow {sow} — {display_name}", fontsize=9)
            if col == 0:
                ax.set_ylabel("Rolling activity")
            if row == 0:
                ax.legend(loc="upper left", fontsize=7)

    for ax in axes[-1, :]:
        ax.set_xlabel("Days relative to onset of farrowing")
    fig.suptitle("Nest-building detection: CUSUM / BEAST / NestDetect")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Accelerometer-side grouped cross-validation performance
# ---------------------------------------------------------------------------

def plot_accelerometer_cv(oof: pd.DataFrame, out_path: Path, bin_hours: float = 12, title_extra: str = ""):
    apply_style()
    oof = oof.copy()
    oof["error_hours"] = oof["y_pred_hours"] - oof["y_true_hours"]

    max_h = max(oof["y_true_hours"].max(), bin_hours)
    edges = np.arange(0, max_h + bin_hours, bin_hours)
    oof["bin"] = pd.cut(oof["y_true_hours"], edges, right=False)

    bins_present = [b for b in oof["bin"].cat.categories if (oof["bin"] == b).any()]
    data = [oof.loc[oof["bin"] == b, "error_hours"].to_numpy() for b in bins_present]
    counts = [len(d) for d in data]
    tick_labels = [f"{int(b.left)}-{int(b.right)}h\n(n={n})" for b, n in zip(bins_present, counts)]
    medians = [np.median(d) for d in data]

    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(data) + 2), 5.5))
    ax.boxplot(
        data, positions=range(len(data)), widths=0.6, patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        boxprops=dict(facecolor=PALETTE[0], alpha=0.65, edgecolor=PALETTE[0]),
        whiskerprops=dict(color=PALETTE[0]), capprops=dict(color=PALETTE[0]),
        flierprops=dict(markeredgecolor=PALETTE[0], markersize=4, alpha=0.5),
    )
    ax.plot(range(len(data)), medians, color=PALETTE[3], marker="o", linewidth=1.5,
             zorder=5, label="median error")
    ax.axhline(0, color="black", linewidth=1, linestyle="--", label="no error")

    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_xlabel("True time to farrowing (binned)")
    ax.set_ylabel("Prediction error [h]\n(predicted − true)")

    overall_mae = oof["error_hours"].abs().mean()
    overall_pcc = np.corrcoef(oof["y_true_hours"], oof["y_pred_hours"])[0, 1] if len(oof) > 1 else float("nan")
    ax.set_title(f"Accelerometer grouped CV — out-of-fold error by time-to-farrowing{title_extra}\n"
                 f"overall MAE={overall_mae:.2f}h, PCC={overall_pcc:.3f}, n={len(oof)}")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Video transfer (restyled) — naive / domain-adapted / per-sow baseline
# ---------------------------------------------------------------------------

def plot_transfer_styled(predictions: pd.DataFrame, out_path: Path):
    apply_style()
    strategy_display = {
        "naive_accelerometer_scaler": ("naive\n(accelerometer scaler)", PALETTE[3]),
        "video_domain_scaler": ("domain-adapted\n(video scaler)", PALETTE[0]),
        "per_sow_baseline": ("per-sow baseline", PALETTE[2]),
    }
    strategies = [s for s in strategy_display if s in predictions["strategy"].unique()]

    fig, axes = plt.subplots(1, len(strategies), figsize=(5 * len(strategies), 5))
    if len(strategies) == 1:
        axes = [axes]
    for ax, strategy in zip(axes, strategies):
        label, color = strategy_display[strategy]
        sub = predictions[predictions["strategy"] == strategy]
        y_true = sub["y_true_hours"].to_numpy()
        pred = sub["y_pred_hours"].to_numpy()

        ax.scatter(y_true, pred, s=16, alpha=0.75, color=color)
        lim = max(y_true.max(), pred.max(), 1) * 1.05
        ax.plot([0, lim], [0, lim], color="black", linewidth=1, linestyle="--")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("Real time to farrowing [h]")
        ax.set_ylabel("Predicted time to farrowing [h]")

        mae = np.abs(y_true - pred).mean()
        pcc = np.corrcoef(y_true, pred)[0, 1] if len(y_true) > 1 else float("nan")
        ax.set_title(f"{label}\nMAE={mae:.2f}h, PCC={pcc:.3f}, n={len(y_true)}")
    fig.suptitle("Accelerometer-trained model applied to video data")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Feature sensitivity — the SHAP alternative
# ---------------------------------------------------------------------------

def plot_feature_sensitivity(sensitivity: pd.DataFrame, out_path: Path):
    """SHAP explains a *black box* by sampling perturbations around a
    background set and attributing the resulting change in output —
    an approximation that exists because you can't read the model's
    formula directly. A fitted Operon model is the opposite case: the
    formula (``model.formula()``) already *is* the exact rule, so
    ``cross_signal.compute_feature_sensitivity`` (run once, right after
    training, in ``sr_transfer.py``) perturbs each feature and reads the
    model's own prediction — the same kind of "effect of this feature"
    curve SHAP would estimate, but exact rather than sampled.

    ``sensitivity`` is the long-format DataFrame
    ``compute_feature_sensitivity`` returns (``feature``, ``x``,
    ``y_pred_hours``, ``reference_value``) — this function only plots it.
    """
    apply_style()
    features = sensitivity["feature"].unique().tolist()
    ncols = 5
    nrows = -(-len(features) // ncols)  # ceil

    fig, axes = plt.subplots(nrows, ncols, figsize=(17, 3.4 * nrows), squeeze=False)
    axes = axes.ravel()
    for i, feat in enumerate(features):
        ax = axes[i]
        sub = sensitivity[sensitivity["feature"] == feat]
        ax.plot(sub["x"], sub["y_pred_hours"], color=PALETTE[0], linewidth=1.8)
        ax.axvline(sub["reference_value"].iloc[0], color=PALETTE[3], linewidth=1, linestyle=":", alpha=0.8)
        ax.set_title(feat, fontsize=9)
        ax.tick_params(labelsize=7)
        if i % ncols == 0:
            ax.set_ylabel("Predicted time to\nfarrowing [h]", fontsize=8)
    for j in range(len(features), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Feature sensitivity — exact effect curves read off the fitted formula\n"
                 "(other features held at their training-set median; dotted line = that median)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Shared scatter-panel helper for figures 6-7
# ---------------------------------------------------------------------------

def _scatter_panel(ax, groups: list, title: str, fallback_text: str):
    groups = [(yt, yp, c, l) for yt, yp, c, l in groups if len(yt) > 0]
    ax.set_title(title, fontsize=9)
    if not groups:
        ax.text(0.5, 0.5, fallback_text, ha="center", va="center", transform=ax.transAxes, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        return

    for y_true, y_pred, color, label in groups:
        ax.scatter(y_true, y_pred, s=16, alpha=0.7, color=color, label=label)
    lim = max(max(g[0].max() for g in groups), max(g[1].max() for g in groups), 1) * 1.05
    ax.plot([0, lim], [0, lim], color="black", linewidth=1, linestyle="--")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    if len(groups) > 1:
        ax.legend(loc="upper left", fontsize=7)


# ---------------------------------------------------------------------------
# 6. Farrowing-anchored prediction scatter, per named animal
# ---------------------------------------------------------------------------

def plot_anchored_prediction_scatter(sows: list, predictions: pd.DataFrame, out_path: Path):
    apply_style()
    fig, axes = plt.subplots(len(sows), 2, figsize=(11, 3.6 * len(sows)), squeeze=False)

    for row, sow in enumerate(sows):
        sow_data = predictions[predictions["sow"] == sow]

        acc = sow_data[(sow_data["signal"] == "accelerometer")
                        & (sow_data["scaling_strategy"] == "native_accelerometer_scaler")]
        _scatter_panel(axes[row, 0],
                       [(acc["y_true_hours"].to_numpy(), acc["y_pred_hours"].to_numpy(), PALETTE[1], "accelerometer")],
                       f"Sow {sow} — accelerometer", "no accelerometer data")
        axes[row, 0].set_ylabel("Predicted time to\nfarrowing [h]")

        video_groups = [
            (sow_data[(sow_data["signal"] == "video") & (sow_data["scaling_strategy"] == strategy)]
             ["y_true_hours"].to_numpy(),
             sow_data[(sow_data["signal"] == "video") & (sow_data["scaling_strategy"] == strategy)]
             ["y_pred_hours"].to_numpy(),
             color, label)
            for strategy, label, color in VIDEO_STRATEGIES_FOR_SCATTER
        ]
        _scatter_panel(axes[row, 1], video_groups, f"Sow {sow} — video", "no video data")

    for ax in axes[-1, :]:
        ax.set_xlabel("Real time to farrowing [h]")
    fig.suptitle("Time-to-farrowing prediction (farrowing-anchored)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Online-setting prediction scatter, per named animal
# ---------------------------------------------------------------------------

def plot_online_prediction_scatter(sows: list, predictions: pd.DataFrame, out_path: Path,
                                    method: str = "NestDetect"):
    apply_style()
    fig, axes = plt.subplots(len(sows), 2, figsize=(11, 3.6 * len(sows)), squeeze=False)
    no_window_text = f"no online window\n({method} didn't fire\nbefore farrowing)"

    for row, sow in enumerate(sows):
        sow_data = predictions[(predictions["sow"] == sow) & (predictions["method"] == method)]

        acc = sow_data[(sow_data["signal"] == "accelerometer")
                        & (sow_data["scaling_strategy"] == "native_accelerometer_scaler")]
        _scatter_panel(axes[row, 0],
                       [(acc["y_true_hours"].to_numpy(), acc["y_pred_hours"].to_numpy(), PALETTE[1], "accelerometer")],
                       f"Sow {sow} — accelerometer", no_window_text)
        axes[row, 0].set_ylabel("Predicted time to\nfarrowing [h]")

        video_groups = [
            (sow_data[(sow_data["signal"] == "video") & (sow_data["scaling_strategy"] == strategy)]
             ["y_true_hours"].to_numpy(),
             sow_data[(sow_data["signal"] == "video") & (sow_data["scaling_strategy"] == strategy)]
             ["y_pred_hours"].to_numpy(),
             color, label)
            for strategy, label, color in VIDEO_STRATEGIES_FOR_SCATTER
        ]
        _scatter_panel(axes[row, 1], video_groups, f"Sow {sow} — video", no_window_text)

    for ax in axes[-1, :]:
        ax.set_xlabel("Real time to farrowing [h]")
    fig.suptitle(f"Time-to-farrowing prediction (online setting, {method}-triggered)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration — reads already-written results, trains/detects nothing
# ---------------------------------------------------------------------------

def generate_all(sows: list, results_dir: Path = None, named_sow_eval_dir: Path = None,
                  online_method: str = "NestDetect"):
    results_dir = results_dir or SR_TRANSFER_DIR
    named_sow_eval_dir = named_sow_eval_dir or NAMED_SOW_EVAL_DIR
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Figure 1/2 sows: {sows}")
    plot_signal_comparison(sows, OUT_DIR / "1_signal_comparison.png")
    print(f"Wrote {OUT_DIR}/1_signal_comparison.png")
    plot_nest_building_comparison(sows, OUT_DIR / "2_nest_building_comparison.png")
    print(f"Wrote {OUT_DIR}/2_nest_building_comparison.png")

    oof_path = results_dir / "accelerometer_cv_oof.csv"
    if oof_path.exists():
        plot_accelerometer_cv(pd.read_csv(oof_path), OUT_DIR / "3_accelerometer_cv.png")
        print(f"Wrote {OUT_DIR}/3_accelerometer_cv.png (from {oof_path})")
    else:
        print(f"{oof_path} not found — skipping figure 3. Run `python -m video_activity.sr_transfer` "
              f"first (needs >=2 accelerometer sows for grouped CV to produce this file).")

    transfer_path = results_dir / "transfer_predictions.csv"
    if transfer_path.exists():
        plot_transfer_styled(pd.read_csv(transfer_path), OUT_DIR / "4_transfer.png")
        print(f"Wrote {OUT_DIR}/4_transfer.png (from {transfer_path})")
    else:
        print(f"{transfer_path} not found — skipping figure 4. Run `python -m video_activity.sr_transfer` "
              f"first (needs at least one video-paired sow to produce this file).")

    sensitivity_path = results_dir / "feature_sensitivity.csv"
    if sensitivity_path.exists():
        plot_feature_sensitivity(pd.read_csv(sensitivity_path), OUT_DIR / "5_feature_sensitivity.png")
        print(f"Wrote {OUT_DIR}/5_feature_sensitivity.png (from {sensitivity_path})")
    else:
        print(f"{sensitivity_path} not found — skipping figure 5. Run "
              f"`python -m video_activity.sr_transfer` first.")

    anchored_path = named_sow_eval_dir / "anchored_predictions.csv"
    if anchored_path.exists():
        plot_anchored_prediction_scatter(sows, pd.read_csv(anchored_path), OUT_DIR / "6_anchored_prediction.png")
        print(f"Wrote {OUT_DIR}/6_anchored_prediction.png (from {anchored_path})")
    else:
        print(f"{anchored_path} not found — skipping figure 6. Run "
              f"`python -m video_activity.named_sow_eval --sow {' '.join(repr(s) for s in sows)}` first.")

    online_path = named_sow_eval_dir / "online_predictions.csv"
    if online_path.exists():
        plot_online_prediction_scatter(sows, pd.read_csv(online_path), OUT_DIR / "7_online_prediction.png",
                                        method=online_method)
        print(f"Wrote {OUT_DIR}/7_online_prediction.png (from {online_path})")
    else:
        print(f"{online_path} not found — skipping figure 7. Run "
              f"`python -m video_activity.named_sow_eval --sow {' '.join(repr(s) for s in sows)}` first.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sows", nargs="+", default=None,
                         help="Sows for figures 1-2 (signal comparison, detection). "
                              "Defaults to every complete accelerometer+video+meta pair found, capped at 3.")
    parser.add_argument("--results-dir", type=str, default=None,
                         help="Where to read figures 3-5's input CSVs from. "
                              "Defaults to out/video_case_study/sr_transfer (sr_transfer.py's output dir).")
    parser.add_argument("--named-sow-eval-dir", type=str, default=None,
                         help="Where to read figures 6-7's input CSVs from. Defaults to "
                              "out/video_case_study/named_sow_eval (named_sow_eval.py's output dir).")
    parser.add_argument("--online-method", type=str, default="NestDetect",
                         choices=["NestDetect", "CUSUM", "BEAST"],
                         help="Which detector's online result figure 7 shows.")
    args = parser.parse_args()

    sows = args.sows
    if sows is None:
        sows = suite.complete_pairs()[:3]
        if not sows:
            raise SystemExit("No complete accelerometer+video+meta pairs found under data/raw/ — "
                              "pass --sows explicitly, or add data for at least one sow.")

    generate_all(
        sows,
        results_dir=Path(args.results_dir) if args.results_dir else None,
        named_sow_eval_dir=Path(args.named_sow_eval_dir) if args.named_sow_eval_dir else None,
        online_method=args.online_method,
    )
