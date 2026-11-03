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

"""Per-instance prediction results — farrowing-anchored and online.

Unlike sr_transfer.py's pooled grouped-CV/transfer numbers (every sow
pooled together into one aggregate) or online_eval.py's aggregated
MAE/PCC per sow x signal x method, this keeps every individual
instance's true vs. predicted value, specifically so a scatter plot can
show one chosen animal's own prediction trajectory rather than a
population summary.

Usage (from the repo root):

    python -m video_activity.sr_transfer     # trains + persists the model, once
    python -m video_activity.named_sow_eval --sow "147002 7" "147037 18" "147145 2"

Output (``out/video_case_study/named_sow_eval/``):

- ``anchored_predictions.csv`` — farrowing-anchored window
  (``instance_tables.build_instance_table``): sow, signal,
  scaling_strategy, y_true_hours, y_pred_hours.
- ``online_predictions.csv`` — online window
  (``instance_tables.build_online_instance_table``, only where a detector
  fired before farrowing): sow, signal, method, scaling_strategy,
  y_true_hours, y_pred_hours.
"""
import argparse
from pathlib import Path

import pandas as pd

from video_activity import suite
from video_activity.online_eval import (
    LOADERS, METHODS, SR_TRANSFER_DIR, load_selected_model, _scaled_predictions,
    evaluate_online_for_sow_signal,
)
from video_activity.ingestion import load_meta
from farrowing_prediction.training.instance_tables import build_instance_table

OUT_DIR = suite.case_study.OUT_DIR / "named_sow_eval"


def anchored_predictions_for_sow(sow: str, model, accelerometer_scaler, per_sow_baselines: dict) -> pd.DataFrame:
    """Farrowing-anchored per-instance predictions for every signal this
    sow has, under every applicable scaling strategy. Skips a signal the
    sow doesn't have data for or that errors (e.g. too little
    pre-farrowing history), rather than failing the whole sow.
    """
    meta = load_meta(sow)
    farrowing_time = meta["start_farrowing"]
    close_crate = meta["close_crate"] if pd.notna(meta["close_crate"]) else None
    open_crate = meta["open_crate"] if pd.notna(meta["open_crate"]) else None

    rows = []
    for signal_name, loader in LOADERS.items():
        try:
            series = loader(sow)
            table = build_instance_table(series, farrowing_time, close_crate, open_crate)
            table["sow"] = sow
            scored = _scaled_predictions(table, signal_name, sow, model, accelerometer_scaler,
                                          per_sow_baselines, return_rows=True)
        except Exception as e:  # one bad signal must not cost the other
            print(f"  anchored prediction failed for {sow}/{signal_name}: {type(e).__name__}: {e}")
            continue

        for strategy, res in scored.items():
            rows.append(pd.DataFrame({
                "sow": sow, "signal": signal_name, "scaling_strategy": strategy,
                "y_true_hours": res["y_true_hours"], "y_pred_hours": res["y_pred_hours"],
            }))

    if not rows:
        return pd.DataFrame(columns=["sow", "signal", "scaling_strategy", "y_true_hours", "y_pred_hours"])
    return pd.concat(rows, ignore_index=True)


def online_predictions_for_sow(sow: str, model, accelerometer_scaler, per_sow_baselines: dict,
                                methods=METHODS) -> pd.DataFrame:
    """Online per-instance predictions for every signal this sow has,
    across every method in ``methods`` that actually fired before
    farrowing (one that didn't contributes no rows here — see
    online_eval.py's own report for *why* it didn't)."""
    rows = []
    for signal_name in LOADERS:
        try:
            results = evaluate_online_for_sow_signal(
                sow, signal_name, model, accelerometer_scaler, per_sow_baselines,
                methods=methods, return_rows=True)
        except Exception as e:  # one bad signal must not cost the other
            print(f"  online prediction failed for {sow}/{signal_name}: {type(e).__name__}: {e}")
            continue

        for res in results:
            if res["online_eval"] is None:
                continue
            for strategy, strat_res in res["online_eval"].items():
                rows.append(pd.DataFrame({
                    "sow": sow, "signal": signal_name, "method": res["method"], "scaling_strategy": strategy,
                    "y_true_hours": strat_res["y_true_hours"], "y_pred_hours": strat_res["y_pred_hours"],
                }))

    if not rows:
        return pd.DataFrame(columns=["sow", "signal", "method", "scaling_strategy",
                                      "y_true_hours", "y_pred_hours"])
    return pd.concat(rows, ignore_index=True)


def run(sows: list, methods=METHODS, sr_transfer_dir: Path = SR_TRANSFER_DIR):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, accelerometer_scaler, per_sow_baselines = load_selected_model(sr_transfer_dir)

    anchored = pd.concat(
        [anchored_predictions_for_sow(sow, model, accelerometer_scaler, per_sow_baselines) for sow in sows],
        ignore_index=True,
    )
    online = pd.concat(
        [online_predictions_for_sow(sow, model, accelerometer_scaler, per_sow_baselines, methods=methods)
         for sow in sows],
        ignore_index=True,
    )

    anchored.to_csv(OUT_DIR / "anchored_predictions.csv", index=False)
    online.to_csv(OUT_DIR / "online_predictions.csv", index=False)
    print(f"Wrote {OUT_DIR}/anchored_predictions.csv ({len(anchored)} rows), "
          f"online_predictions.csv ({len(online)} rows)")
    return {"sows": sows, "anchored": anchored, "online": online}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sow", nargs="+", required=True, dest="sows", help="Sows to evaluate.")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS),
                         help="Which detector(s) to use as the online trigger.")
    args = parser.parse_args()
    run(sows=args.sows, methods=tuple(args.methods))
