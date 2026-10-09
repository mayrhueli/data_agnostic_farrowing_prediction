# Video-Derived Activity Farrowing Prediction

Extends an interpretable, two-stage accelerometer-based farrowing
prediction approach (nest-building detection, then time-to-farrowing
prediction via symbolic regression) to video-derived (optical-flow)
activity data, and tests whether a model trained on accelerometer data
generalizes to video — with three different cross-signal scaling
strategies, a "farrowing-anchored" evaluation and a deployment-realistic
"online" one (detection-triggered, walk-forward prediction), and a
styled figure set for all of it.

## Installation

```bash
pip install -r requirements.txt
```

`Rbeast` and `pyoperon` are the two dependencies most likely to need a
platform-specific install step (compiled wheels). If either fails,
nest-building detection still works via CUSUM/NestDetect without
`Rbeast` (`run_detection_methods` treats BEAST as unavailable and
reports `None` for it instead of failing the sow), but symbolic
regression (`sr_transfer.py`, `online_eval.py`, `named_sow_eval.py`)
needs `pyoperon`.


## Data layout

```
data/raw/accelerometer/<sow>.csv    # timestamp,Acc            (1-minute magnitude)
data/raw/video/<sow>_activity.csv   # Timestamp,Flow_totals    (~1 Hz optical-flow activity)
data/raw/meta/<sow>.csv             # one row: farrowing/crate timestamps, pen type
```

`<sow>` is a sow identifier matching the filenames, used as-is
throughout (e.g. `100001 10`).

`data/` is gitignored — bring your own and drop new sows in following
this layout; they're picked up automatically.

## Running one sow

```bash
python -m video_activity.case_study --sow "100001 10"
```

Runs nest-building detection (CUSUM, BEAST, NestDetect) on both signals
and writes `out/video_case_study/<sow>_detection.png` and
`<sow>_summary.json`.

## Running the suite (every pair you have)

```bash
python -m video_activity.suite                              # every complete triplet found
python -m video_activity.suite --sow "100001 10" --sow "X"   # restrict to specific sows
```

`suite.py` scans `data/raw/accelerometer/`, `data/raw/video/` and
`data/raw/meta/`, runs `case_study.main()` for every sow that has all three
files, and aggregates the detection results:

- a sow missing any of the three files is **skipped** and listed, not
  treated as an error;
- a sow that raises while processing (e.g. too little data before
  farrowing, unparsable timestamps) is **caught and reported**, so one
  bad pair doesn't stop the rest of the suite;
- outputs land in `out/video_case_study/suite/`:
  - `detection_long.csv` — one row per sow x signal x method;
  - `detection_summary.csv` — aggregated across sows (% early/on-time/late
    per method+signal);
  - `detection_summary.png` — a bar chart of the above;
  - `suite_report.md` — a short auto-generated summary (skipped/failed
    sows + the aggregate table);
  - `suite_results.json` — which sows ran, which were skipped, which
    errored and why.

### Adding more pairs

Drop each new sow in following the layout above and re-run the suite —
nothing else needs to change. `pytest video_activity/test/test_suite.py`
also parametrizes its real-data test over whatever complete pairs it
finds, so it automatically covers new sows too; the rest of that file's
tests (discovery, aggregation math, error isolation) run against
synthetic data and don't need any real files.

## Symbolic regression (Operon): train on accelerometer, test on video

```bash
python -m video_activity.sr_transfer                                    # defaults: population 1000, generations 200
python -m video_activity.sr_transfer --population-size 500 --generations 150
```

This is the headline method: instead of a fresh model per sow, it pools
every sow's accelerometer data (`accelerometer_sows()` — needs metadata
+ accelerometer, video not required), trains one Operon symbolic-regression
model (`farrowing_prediction/training/operon_sr.py`, `.../cross_signal.py`)
with grouped k-fold CV by sow once ≥2 sows are available — with only one
it falls back to a single-sow train/test smoke test and says so — and
then applies that *same* model, completely unmodified, to every
video-paired sow to see whether it generalizes across signal types.

**Video-paired sows are excluded from the accelerometer training pool by
default** (`run(..., exclude_video_sows_from_training=True)`), so the
model doing the video-transfer check has never seen those animals at
all, not just their video data — without this, since accelerometer
coverage is a superset of video coverage, grouped CV's "best fold" model
would typically have been trained on most of the same sows it's then
"tested" on (via their accelerometer trace instead of video), quietly
turning "generalizes to a new animal" into "generalizes to a new signal
for an animal it partly trained on." `summary["held_out_transfer_eval"]`
(and a `[held-out ...]`/`[NOT held-out ...]` label on the printed
transfer results) says which case applies; if excluding video sows would
leave nothing to train on it falls back to including them rather than
crashing, and says so. Pass `--include-video-sows-in-training` to opt
out on purpose.

Accelerometer and video are ~1e5-1e6x apart in raw magnitude, so the eight
activity-magnitude features are rescaled before the model sees them, under
three strategies, all reported together:

- **naive**: reuse the accelerometer-fitted `StandardScaler` on raw video
  values — physically meaningless (z-scores computed from one unit system
  applied to values in a completely different one).
- **domain-level**: a fresh `StandardScaler` fit on the video table's own
  (unlabelled) feature statistics. Needs the whole video population's
  statistics in advance — including, in this evaluation, the very sows
  being tested — so it's a hindsight adaptation, not something available
  for a genuinely new animal on day one.
- **per-sow baseline**: each sow's activity features divided (not
  z-scored — a ratio to that animal's own resting level) by *that sow's
  own* mean activity in its first 48h after pen entry, before
  nest-building typically begins. Needs only one animal's own early data,
  so — unlike domain-level — it's usable online for a brand-new sow.
  `sr_transfer.compute_per_sow_baselines()` (`video_activity/pooling.py`)
  computes it straight from each sow's raw series and metadata; a sow
  with no data in that window (e.g. video started late) is skipped for
  this strategy specifically — reported under
  `per_sow_baseline_errors` in the summary — while still contributing to
  the naive/domain-level results, which don't need it. The window is
  configurable (`--baseline-window-hours`, default 48).

The two crating features are never rescaled under any strategy, since
they mean the same thing regardless of sensor.

`operon_sr.py`'s defaults are deliberately **not** `pyoperon`'s own —
`SymbolicRegressor`'s library defaults disable local coefficient search
(`optimizer_iterations=0`, silently off) and use single-objective fitness
with no parsimony pressure (`objectives=['r2']`), confirmed by reading its
source rather than by reputation. Overridden here to
`optimizer_iterations=10`, `objectives=['r2', 'length']` (NSGA2), smaller
`max_length`/`max_depth` (25/7 vs. 50/10), and `aq` (analytic quotient) in
place of plain `div` (which blows up near a zero denominator — part of
why naive transfer can overflow numerically). All tunable from the CLI
(`--optimizer-iterations`, `--max-length`, `--max-depth`,
`--single-objective`, `--allowed-symbols`).

Output: `out/video_case_study/sr_transfer/sr_transfer_summary.json` and
`transfer_scatter.png`, plus the per-row CSVs described below (for
`figures.py`).

## Styled figure set (`figures.py`)

```bash
python -m video_activity.case_study --sow "100001 10"           # writes <sow>_summary.json (figure 2 needs it)
python -m video_activity.sr_transfer                             # writes the CSVs figures 3-5 need
python -m video_activity.named_sow_eval --sow "100001 10" "A" "B"  # writes the CSVs figures 6-7 need
python -m video_activity.figures --sows "100001 10" "A" "B"     # pick your 3 (or however many) named sows
```

**`figures.py` only reads and plots — it never retrains a model or
reruns detection.** Figures 3-5 read the CSVs `sr_transfer.py` writes
alongside its JSON summary (`accelerometer_cv_oof.csv`,
`transfer_predictions.csv`, `feature_sensitivity.csv`); figures 6-7 read
`named_sow_eval.py`'s per-instance predictions (`anchored_predictions.csv`,
`online_predictions.csv`); figure 2 reads each sow's `<sow>_summary.json`
from `case_study.py`/`suite.py` instead of rerunning
CUSUM/BEAST/NestDetect. This matters for two reasons: a fresh Operon run
isn't reproducible (so redrawing a figure by retraining could quietly
show different numbers than whatever's in your written
`sr_transfer_summary.json`), and BEAST specifically is stochastic (MCMC),
so rerunning it would draw a *different* changepoint than the one
already reported. A missing input file fails with a clear message
naming the command that produces it, rather than silently recomputing
it. Figure 1 is the exception — it reads raw sensor files directly,
since there's no "result" to reuse there, just the deterministic input
data.

Seven figures, all using one fixed color palette (`#4E79A7 #F28E2B #59A14F
#E15759 #B07AA1 #76B7B2`) applied only to plotted content — every label,
tick, title and legend entry stays black — so a color means the same
thing across every plot, written to `out/video_case_study/figures/`:

1. `1_signal_comparison.png` — accelerometer vs. video activity (12h
   rolling mean, dual y-axis for the ~1e5x unit difference), one row per
   sow given via `--sows`.
2. `2_nest_building_comparison.png` — the same sows, activity with each
   detector's (CUSUM/BEAST/NestDetect) changepoint and the true farrowing
   onset marked, accelerometer and video side by side.
3. `3_accelerometer_cv.png` — out-of-fold prediction error (predicted −
   true), binned by the true time-to-farrowing and drawn as one boxplot
   per bin, pooled across the accelerometer grouped CV (every point's
   prediction came from a model that never saw that sow in training). A
   plain true-vs-predicted scatter puts every instance in one cloud and
   reads as noise once accuracy is only moderate; binning by horizon
   instead answers the actually operational question — is the model more
   or less accurate as farrowing approaches — and a median-error line
   away from zero flags systematic bias at a glance. Needs ≥2
   accelerometer sows/groups, same rule as `sr_transfer.py`; if
   `accelerometer_cv_oof.csv` isn't there (single-sow smoke-test mode),
   this figure is skipped with a message rather than faked.
4. `4_transfer.png` — the video-transfer scatter (naive / domain-adapted /
   per-sow baseline), restyled; same numbers `sr_transfer.py` reports,
   since it's built from the exact same per-row predictions.
5. `5_feature_sensitivity.png` — one curve per feature instead of a SHAP
   plot: see the function's docstring for why. Short version — Operon
   already hands you the exact formula (`model.formula()`), so there's no
   black box to approximate; varying one feature at a time through the
   formula while holding the rest at their median gives the same kind of
   "effect of this feature" reading SHAP would estimate, but exact rather
   than sampled (computed once in `sr_transfer.py` right after training,
   via `cross_signal.compute_feature_sensitivity`).
6. `6_anchored_prediction.png` — true vs. predicted time-to-farrowing,
   **farrowing-anchored**, one row per sow given via `--sows`,
   accelerometer | video. The video panel overlays domain-adapted and
   per-sow baseline (comparable scales); naive is left out — figure 4
   already shows it's off by several orders of magnitude, so including
   it here would crush the other two to invisible on any sane axis.
7. `7_online_prediction.png` — the same layout and scaling choices as
   figure 6, but in the **online** setting (`--online-method`, default
   NestDetect — see "Online evaluation" below): a sow/signal where that
   detector never fired before farrowing shows that explicitly as a
   panel of text, not a blank or missing subplot, since that's a real
   result, not a gap to paper over.

Figures 1-2 render for exactly the sows you pass, and need that sow's
`<sow>_summary.json` for figure 2. Pass `--results-dir` / `--named-sow-eval-dir`
to point figures 3-5 / 6-7 at different output directories than the
defaults (`out/video_case_study/sr_transfer`,
`out/video_case_study/named_sow_eval`).

## Recording-gap report (`gap_report.py`)

```bash
python -m video_activity.gap_report                       # every sow with accelerometer or video data
python -m video_activity.gap_report --sow "100001 10" "X"  # restrict to specific sows
```

Worth running before trusting a detection or prediction result — a gap
sitting right before farrowing can explain a missed changepoint or a
noisy instance far more plausibly than a model failure. For every sow
and signal, finds every jump between consecutive valid readings bigger
than 1.5x the expected 1-minute interval, and reports:

- **number** — `n_gaps` per sow x signal.
- **total hours** — `total_gap_hours` (plus `largest_gap_hours`,
  `mean_gap_hours`).
- **position relative to farrowing** — each individual gap's
  `hours_before_farrowing_start`/`_end` (positive = that many hours
  before farrowing onset, negative = after) and a `position` label
  (`"before farrowing"`, `"after farrowing"`, or `"spans farrowing"`).

Accelerometer gaps show up as literally missing rows; video gaps show up
as `NaN` runs instead, since `load_video_series` resamples onto a
complete 1-minute grid and fills missing minutes with `NaN` rather than
dropping them. `detect_gaps()` handles both the same way (drop `NaN` Acc
rows, then look at the gaps between what's left), so one function covers
both signal types without caring which is which. A sow with no usable
metadata still gets its gap counts (just not positioned relative to
farrowing, since there's no farrowing time to position against) —
reported under `meta_error`, not dropped from the run.

Output (`out/video_case_study/gap_report/`): `gap_report_long.csv` (one
row per gap), `gap_report_summary.csv` (one row per sow x signal),
`gap_report.md` (summary table plus every gap within 48h of farrowing,
either side — the ones most likely to matter for a result).

## Online evaluation (`online_eval.py`): the full pipeline, chained

Everything above tests detection and prediction **separately**.
`case_study.py`/`suite.py` run detection on both signals but never feed
it into a prediction step; `sr_transfer.py`'s video-transfer numbers use
`build_instance_table`'s **farrowing-anchored** window (the last 48h
before the *known* farrowing time) regardless of what any detector found
— which needs to already know when farrowing happens to decide where
the window starts, so it isn't a deployment-realistic test either.
`online_eval.py` is the piece that was missing: detection first, and
prediction only starts (and keeps running, hourly, using only data
available at each point) once a detector actually raises an alarm —
exactly what a live system would do.

```bash
python -m video_activity.sr_transfer                     # trains + persists the model (once)
python -m video_activity.online_eval                      # every sow with accelerometer or video data
python -m video_activity.online_eval --sow "100001 10"     # restrict to specific sows
python -m video_activity.online_eval --methods NestDetect  # just one detector as the trigger
```

Uses the **selected model** — the one `sr_transfer.py` actually trained
and persisted — not a freshly fit stand-in, for both accelerometer and
video. For each sow x signal x detector:

1. Run that detector (CUSUM/BEAST/NestDetect) up to 12h past farrowing.
2. If it never fires, or fires at/after farrowing onset, there is **no**
   online prediction window — recorded as a result (`online_eval_error`),
   not skipped. A farrowing-anchored prediction-only test can't see this
   failure mode at all, since it always has a window by construction;
   catching it is the actual point of this module.
3. Otherwise, slide hourly windows from the alarm time to farrowing
   (`instance_tables.build_online_instance_table` — same per-row feature
   extraction as `build_instance_table`, using only `df.iloc[:i]` at
   each step, just windowed from when the alarm fired instead of a fixed
   offset before the already-known farrowing time) and score the
   selected model against it: the model's own (native) scaler for
   accelerometer, and naive/domain-adapted/per-sow for video — the same
   three strategies `evaluate_transfer` reports, so the online numbers
   are directly comparable to the farrowing-anchored transfer ones.

`held_out` on each row says whether the selected (best-fold) model ever
saw that sow during training, derived from `accelerometer_cv_oof.csv` +
`sr_transfer_summary.json` (no retraining needed) rather than assumed —
`None`/blank if that can't be determined (e.g. the single-sow
smoke-test fallback has no fold structure to derive it from).

**Model persistence note**: pyoperon's fitted model object
(`pyoperon.pyoperon.Tree`) cannot be pickled in this build — its
`__getstate__` returns a type pybind11 fails to convert back to Python,
confirmed by hitting it directly rather than assumed. `sr_transfer.py`
instead persists the model's **formula string** (`model.json`), which
`operon_sr.evaluate_formula`/`FormulaModel` evaluate directly (verified
to reproduce `.predict()` to ~1e-6) — the exact fitted function, just
without the population/search state inference doesn't need anyway.

Output (`out/video_case_study/online_eval/`): `online_eval_long.csv`
(one row per sow x signal x method x scaling strategy — `mae_hours`/
`pcc` null where no window existed, with the reason in
`online_eval_error`), `online_eval_report.md`.

## Per-animal predictions (`named_sow_eval.py`): the data figures 6-7 need

`sr_transfer.py` and `online_eval.py` only ever report **aggregates**
(pooled across sows, or MAE/PCC per sow x signal x method) — neither
keeps individual instances' true-vs-predicted values, so neither can
feed a scatter plot of one specific animal's own prediction trajectory.
`named_sow_eval.py` is that: the same selected model, the same scoring
logic (farrowing-anchored via `build_instance_table`, online via
`build_online_instance_table` triggered by detection), just keeping
every instance instead of summarizing it away.

```bash
python -m video_activity.sr_transfer                                 # trains + persists the model, once
python -m video_activity.named_sow_eval --sow "100001 10" "100001 11" "100001 12"
```

Output (`out/video_case_study/named_sow_eval/`):

- `anchored_predictions.csv` — sow, signal, scaling_strategy,
  y_true_hours, y_pred_hours (farrowing-anchored).
- `online_predictions.csv` — sow, signal, method, scaling_strategy,
  y_true_hours, y_pred_hours (online; a method that never fired before
  farrowing for a given sow/signal simply contributes no rows there —
  see `online_eval.py`'s own report for *why*, per sow/signal/method).

`video_activity.figures`'s figures 6-7 read these directly — run this
before `figures.py` to get them.

## What's in each module

- `nest_building/cusum.py`, `nest_building/beast.py`, `nest_building/nestdetect.py`:
  the three nest-building detectors — CUSUM (k=2, threshold=150), BEAST
  (via the `Rbeast` package, daily periodicity, ≤1 changepoint), and
  NestDetect (a sliding-window alarm on rolling 12h/24h activity means).
- `nest_building/evaluation.py`: scores a detected changepoint against
  farrowing onset (24h/48h tolerance windows: early / on-time / late).
- `farrowing_prediction/training/instance_tables.py`: builds per-sow
  instance tables (ten features + a time-to-farrowing target) from a
  1-minute activity series, in two windowing modes — farrowing-anchored
  (`build_instance_table`) and online/detection-triggered
  (`build_online_instance_table`) — via
  `feature_engineering.time_features.calculate_features_for_timeseries`.
- `farrowing_prediction/training/operon_sr.py`,
  `farrowing_prediction/training/cross_signal.py`,
  `video_activity/pooling.py`, `video_activity/sr_transfer.py`: the
  pooled-training, grouped-CV-by-sow symbolic-regression approach (using
  Operon), plus the cross-signal domain-scaling/transfer machinery
  described above. `pooling.py` also computes the per-sow baselines (the
  third scaling strategy) straight from each sow's raw series + metadata.
- `video_activity/case_study.py`, `video_activity/suite.py`: per-sow and
  batch nest-building detection comparisons (accelerometer vs. video).
- `video_activity/online_eval.py`, `video_activity/named_sow_eval.py`,
  `video_activity/figures.py`, `video_activity/gap_report.py`: the
  online (detection-triggered) evaluation setting, per-animal
  prediction extraction, the styled figure set, and the recording-gap
  report.

## Known limitations

- `calculate_features_for_timeseries`'s `mean_day`/`mean_night` are
  computed for a fixed calendar day near the start of the recording, not
  a sliding "last 24h" window — they end up constant (or, for a
  short/gappy series, entirely NaN) per sow and contribute ~0 to the
  fitted models. `build_instance_table` forward/back-fills (falling back
  to 0 only if a column is unrecoverably all-NaN) rather than dropping
  rows for it, since dropping used to silently empty a sow's whole table
  on a short/gappy series and crash a scaler fit on it two calls later.
- BEAST is stochastic (MCMC); each run can shift its changepoint.
- A sow can still fail outright if its accelerometer/video recording
  never reaches 48h of history before farrowing, or never reaches the
  farrowing timestamp at all (sensor removed early, or a metadata/data
  mismatch) — these raise a specific `ValueError` explaining which, are
  caught per-sow by the suite, and are listed in `suite_report.md`'s
  "Failed while processing" section rather than silently skipped.

## License

GPL-3.0 — see [LICENSE](LICENSE).
