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

"""The **online** evaluation setting, end to end: run detection
(CUSUM/BEAST/NestDetect) first, and only start the prediction stage once
(and if) a detector actually raises an alarm, building instances forward
from that alarm time using only data available up to each point.

Run that first:

    python -m video_activity.sr_transfer
    python -m video_activity.online_eval                       # every sow with accelerometer or video data
    python -m video_activity.online_eval --sow "sample_sow"      # restrict to specific sows
    python -m video_activity.online_eval --methods NestDetect   # just one detector


Output (``out/video_case_study/online_eval/``): ``online_eval_long.csv``
(one row per sow x signal x method x scaling strategy), a summary CSV,
and ``online_eval_report.md``.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from joblib import load
from sklearn.metrics import mean_absolute_error

from video_activity import suite
from video_activity.ingestion import load_accelerometer_series, load_video_series, load_meta
from video_activity.case_study import crop_series, run_detection_methods
from video_activity.sr_transfer import OUT_DIR as SR_TRANSFER_DIR
from farrowing_prediction.training.instance_tables import FEATURE_COLUMNS, build_online_instance_table
from farrowing_prediction.training.cross_signal import (
    apply_scaler, apply_per_sow_baseline, fit_activity_scaler, _pcc,
)
from farrowing_prediction.training.operon_sr import OperonSymbolicRegressorWrapper

OUT_DIR = suite.case_study.OUT_DIR / "online_eval"
LOADERS = {"accelerometer": load_accelerometer_series, "video": load_video_series}
METHODS = ("NestDetect", "CUSUM", "BEAST")


def load_selected_model(sr_transfer_dir: Path = SR_TRANSFER_DIR):
    """Load the model + accelerometer scaler + video per-sow baselines
    ``sr_transfer.py`` persisted.
    """
    model_path = sr_transfer_dir / "model.json"
    scaler_path = sr_transfer_dir / "accelerometer_scaler.joblib"
    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            f"{model_path} / {scaler_path} not found — run `python -m video_activity.sr_transfer` "
            f"first to train and persist the model this evaluates."
        )
    model = OperonSymbolicRegressorWrapper.load_model(str(model_path))
    scaler = load(scaler_path)

    baselines_path = sr_transfer_dir / "per_sow_baselines.json"
    per_sow_baselines = {}
    if baselines_path.exists():
        with open(baselines_path) as f:
            per_sow_baselines = {k: float(v) for k, v in json.load(f).items()}

    return model, scaler, per_sow_baselines


def accelerometer_training_pool_sows(sr_transfer_dir: Path = SR_TRANSFER_DIR) -> set:
    """Sows the *selected* (best-fold) model actually trained on, derived
    from ``accelerometer_cv_oof.csv`` + ``sr_transfer_summary.json``
    (no retraining needed). A sow absent from the out-of-fold CSV
    entirely was excluded from the pool altogether (e.g. every
    video-paired sow, under ``sr_transfer.py``'s default
    ``exclude_video_sows_from_training=True``); a sow present but
    landing in the *best* fold's own test split wasn't in that
    specific model's training data either, even though some other
    fold's model did train on it.

    Returns an empty set if the needed files aren't there.
    """
    oof_path = sr_transfer_dir / "accelerometer_cv_oof.csv"
    summary_path = sr_transfer_dir / "sr_transfer_summary.json"
    if not oof_path.exists() or not summary_path.exists():
        return set()

    oof = pd.read_csv(oof_path)
    with open(summary_path) as f:
        summary = json.load(f)
    cv_folds = summary.get("cv_folds")
    if oof.empty or not cv_folds:
        return set()

    best_fold = min(cv_folds, key=lambda fr: fr["mae_hours"])["fold"]
    all_sows = set(oof["sow"])
    best_fold_test_sows = set(oof.loc[oof["fold"] == best_fold, "sow"])
    return all_sows - best_fold_test_sows


def _scaled_predictions(table: pd.DataFrame, signal_name: str, sow: str, model,
                         accelerometer_scaler, per_sow_baselines: dict, return_rows: bool = False) -> dict:
    """Score ``table`` under every scaling strategy that applies to
    ``signal_name`` — just the model's own (native) scaler for
    accelerometer; naive/domain-adapted/per-sow for video, matching
    ``cross_signal.evaluate_transfer``'s three strategies, so the online
    numbers are directly comparable to the farrowing-anchored transfer
    numbers already reported there.

    With ``return_rows=True``, each strategy's dict also carries
    ``y_true_hours``/``y_pred_hours`` arrays (every instance, not just
    the aggregate) — for persisting per-row predictions so a scatter
    plot can be drawn later without rerunning detection/prediction.
    """
    results = {}
    if signal_name == "accelerometer":
        strategies = [("native_accelerometer_scaler", apply_scaler(table, accelerometer_scaler))]
    else:
        strategies = [
            ("naive_accelerometer_scaler", apply_scaler(table, accelerometer_scaler)),
            ("video_domain_scaler", apply_scaler(table, fit_activity_scaler(table))),
        ]
        if sow in per_sow_baselines:
            strategies.append(("per_sow_baseline", apply_per_sow_baseline(table, {sow: per_sow_baselines[sow]})))

    for label, scaled in strategies:
        pred = model.predict(scaled[FEATURE_COLUMNS])
        y_true = scaled["remaining_minutes"].to_numpy()
        results[label] = {
            "n_instances": len(scaled),
            "mae_hours": mean_absolute_error(y_true, pred) / 60,
            "pcc": _pcc(y_true, pred),
        }
        if return_rows:
            results[label]["y_true_hours"] = y_true / 60
            results[label]["y_pred_hours"] = pred / 60
    return results


def evaluate_online_for_sow_signal(sow: str, signal_name: str, model, accelerometer_scaler,
                                    per_sow_baselines: dict, methods=METHODS, return_rows: bool = False) -> list:
    """Detect, then predict online, for one sow/signal across every
    method in ``methods``. Returns one dict per method — a method whose
    detector never fires, or fires at/after farrowing, gets a dict with
    ``online_eval=None`` and a reason.
    """
    meta = load_meta(sow)
    farrowing_time = meta["start_farrowing"]
    entry = meta["entry_farrowing_pen"]
    close_crate = meta["close_crate"] if pd.notna(meta["close_crate"]) else None
    open_crate = meta["open_crate"] if pd.notna(meta["open_crate"]) else None

    series = LOADERS[signal_name](sow)
    detection_end = farrowing_time + pd.Timedelta(hours=12)
    det_series = crop_series(series, entry, detection_end)
    detections = run_detection_methods(det_series, signal_name, farrowing_time)

    rows = []
    for method in methods:
        det = detections.get(method)
        base = {
            "sow": sow, "signal": signal_name, "method": method,
            "changepoint": str(det.changepoint) if det and det.changepoint is not None else None,
            "lead_time_hours": det.lead_time_hours if det else None,
            "classification_24h": det.classification_24h if det else None,
            "classification_48h": det.classification_48h if det else None,
        }

        if det is None or det.changepoint is None:
            rows.append({**base, "online_eval": None,
                         "online_eval_error": "detector did not fire — no online prediction window"})
            continue
        if det.changepoint >= farrowing_time:
            rows.append({**base, "online_eval": None,
                         "online_eval_error": "detector fired at/after farrowing — no online prediction window"})
            continue

        try:
            table = build_online_instance_table(series, det.changepoint, farrowing_time, close_crate, open_crate)
            # tag with 'sow' — build_online_instance_table is single-sow (unlike
            # pooling.build_pooled_table), but apply_per_sow_baseline needs this
            # column to look up the right baseline for each row.
            table["sow"] = sow
            online_eval = _scaled_predictions(table, signal_name, sow, model, accelerometer_scaler,
                                               per_sow_baselines, return_rows=return_rows)
            rows.append({**base, "online_eval": online_eval, "online_eval_error": None})
        except Exception as e:  # one bad sow/signal/method must not cost the rest
            rows.append({**base, "online_eval": None, "online_eval_error": f"{type(e).__name__}: {e}"})

    return rows


def to_long_df(all_rows: list, training_pool_sows: set) -> pd.DataFrame:
    records = []
    for row in all_rows:
        held_out = (row["sow"] not in training_pool_sows) if training_pool_sows else None
        base = {k: v for k, v in row.items() if k not in ("online_eval",)}
        base["held_out"] = held_out
        if row["online_eval"] is None:
            records.append({**base, "scaling_strategy": None, "n_instances": None,
                             "mae_hours": None, "pcc": None})
            continue
        for strategy, res in row["online_eval"].items():
            records.append({**base, "scaling_strategy": strategy, **res})
    return pd.DataFrame(records)


def write_report(long_df: pd.DataFrame, out_dir: Path) -> str:
    lines = ["# Online evaluation report", "",
             "Prediction only ran where a detector fired before farrowing — "
             "every other row shows why it couldn't, which a farrowing-anchored "
             "prediction-only test never surfaces at all.", ""]

    no_window = long_df[long_df["online_eval_error"].notna()] if "online_eval_error" in long_df.columns \
        else long_df.iloc[0:0]
    total = len(long_df[["sow", "signal", "method"]].drop_duplicates()) if not long_df.empty else 0
    failed = len(no_window[["sow", "signal", "method"]].drop_duplicates()) if not no_window.empty else 0
    pct = f" ({failed / total:.0%})" if total else ""
    lines.append(f"{total} sow x signal x method combination(s) checked; "
                  f"{failed}{pct} had no online prediction window.")
    lines.append("")

    scored = long_df[long_df["mae_hours"].notna()] if "mae_hours" in long_df.columns else long_df.iloc[0:0]
    if not scored.empty:
        lines.append("## Online MAE / PCC (where a prediction window existed)\n")
        cols = ["sow", "signal", "method", "scaling_strategy", "held_out", "n_instances", "mae_hours", "pcc"]
        try:
            lines.append(scored[cols].sort_values(["signal", "method", "sow"]).to_markdown(index=False))
        except ImportError:
            lines.append("```\n" + scored[cols].to_string(index=False) + "\n```")

    if not no_window.empty:
        lines.append("\n## No online prediction window\n")
        cols = ["sow", "signal", "method", "online_eval_error"]
        try:
            lines.append(no_window[cols].drop_duplicates().to_markdown(index=False))
        except ImportError:
            lines.append("```\n" + no_window[cols].drop_duplicates().to_string(index=False) + "\n```")

    text = "\n".join(lines) + "\n"
    (out_dir / "online_eval_report.md").write_text(text)
    return text


def run(sows: list = None, methods=METHODS, sr_transfer_dir: Path = SR_TRANSFER_DIR):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, accelerometer_scaler, per_sow_baselines = load_selected_model(sr_transfer_dir)
    training_pool_sows = accelerometer_training_pool_sows(sr_transfer_dir)
    if not training_pool_sows:
        print("Could not determine the selected model's training pool (no accelerometer_cv_oof.csv / "
              "cv_folds — single-sow smoke-test run?) — 'held_out' will be reported as unknown (None).")

    pairs = suite.discover_pairs()
    sows = sows if sows is not None else sorted(
        sow for sow, flags in pairs.items() if flags["accelerometer"] or flags["video"])

    all_rows = []
    for sow in sows:
        flags = pairs.get(sow, {})
        for signal_name in LOADERS:
            if flags and not flags.get(signal_name, True):
                continue
            try:
                all_rows.extend(evaluate_online_for_sow_signal(
                    sow, signal_name, model, accelerometer_scaler, per_sow_baselines, methods=methods))
            except Exception as e:  # noqa: BLE001 - one bad sow/signal must not cost the rest
                all_rows.append({"sow": sow, "signal": signal_name, "method": None,
                                  "changepoint": None, "lead_time_hours": None,
                                  "classification_24h": None, "classification_48h": None,
                                  "online_eval": None, "online_eval_error": f"{type(e).__name__}: {e}"})

    long_df = to_long_df(all_rows, training_pool_sows)
    long_df.to_csv(OUT_DIR / "online_eval_long.csv", index=False)
    report_text = write_report(long_df, OUT_DIR)

    print(report_text)
    print(f"Wrote {OUT_DIR}/online_eval_long.csv, online_eval_report.md")
    return {"sows": sows, "long_df": long_df}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sow", nargs="+", default=None, dest="sows",
                         help="Restrict to specific sows. Defaults to every sow with an "
                              "accelerometer or video file.")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS),
                         help="Which detector(s) to use as the online trigger.")
    args = parser.parse_args()
    run(sows=args.sows, methods=tuple(args.methods))
