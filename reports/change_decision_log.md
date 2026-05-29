# Program Change Decision Log

This is a chronological record of the main code changes made to `/Users/francis/Desktop/ML_test`, why each change was made, and the logic behind it.

## 0. Baseline Before These Changes

The project already implemented an S&P 500 High/Low forecasting pipeline:

1. Load S&P 500 OHLC data.
2. Split into fit / validation / test.
3. Decompose each target with ICEEMDAN.
4. Re-decompose `IMF1` with PSO-VMD.
5. Train one neural model per component.
6. Sum component predictions into `PredictedHigh` or `PredictedLow`.

The core neural block was:

```text
BiLSTM -> SAM attention -> TCN -> dense head
```

The current config keeps:

```yaml
experiment.window_size: 3
decomposition.scope: walk_forward
decomposition.mode: iceemdan_pso_vmd
model.variant: proposed
features.enabled: true
features.use_for_res: false
```

## Latest. Replace Absolute-Error ACI With Signed One-Sided ACI

The earlier ACI layer used:

```text
score_t = |Actual_t - Predicted_t| / Volatility_t
```

That produced a symmetric conformal width, then the plot displayed only the relevant side for High or Low. This was conservative for Low when the point model already underestimated the target, because absolute error removed the direction of the model bias.

The current version uses signed standardized residuals:

```text
r_t = (Actual_t - Predicted_t) / Volatility_t
```

High and Low are calibrated as separate one-sided bounds:

```text
q_high,t = upper-tail quantile of recent r_t
HighBound_t = PredictedHigh_t + q_high,t * Volatility_t
ErrHigh_t = 1{ActualHigh_t > HighBound_t}

q_low,t = lower-tail quantile of recent r_t
LowBound_t = PredictedLow_t + q_low,t * Volatility_t
ErrLow_t = 1{ActualLow_t < LowBound_t}
```

This keeps the ACI wrapper idea from Gibbs and Candes (2021), but changes the conformity score to preserve whether the point model is underpredicting or overpredicting. The default target coverage is now `0.95`, so HighBound and LowBound are interpreted as separate 95% one-sided bounds, not a single two-sided 95% interval.

## Latest. Export Coverage Between Predicted And Bound

The ACI `Covered` column answers the one-sided conformal question:

```text
High: ActualHigh <= HighBound
Low:  ActualLow >= LowBound
```

The project now also exports the shaded-band diagnostic requested by the user:

```text
BoundBandCovered_t = 1{Actual_t lies between Predicted_t and HighBound_t / LowBound_t}
bound_band_coverage = mean(BoundBandCovered_t)
```

This is written per row in `predictions_high.csv` / `predictions_low.csv`, and the aggregate ratio is stored in `summary_high.json` / `summary_low.json` under `conformal.bound_band_coverage`.

## 1. Add ACI Conformal Boundary Instead Of Raw Error Quantile

### Problem

The original boundary idea was:

```text
HighBound = PredHigh + q * volatility_t
```

But if `q` is estimated from raw errors, a few extreme price-level errors can dominate the boundary. The user pointed out that the error should be standardized first:

```text
Error_t / Volatility_t
```

### Reasoning

The model predicts price levels, but S&P 500 volatility changes over time. A 60-point error during a quiet period and a 60-point error during a volatile period should not have the same calibration meaning.

So the conformal score should be scale-normalized:

```text
score_t = |Actual_t - Predicted_t| / Volatility_t
```

Then the quantile `q` is dimensionless, and the final boundary returns to price units:

```text
HighBound_t = PredictedHigh_t + q_t * Volatility_t
LowBound_t = PredictedLow_t - q_t * Volatility_t
```

### Code Changes

Added:

- `src/sp500_forecast/conformal.py`
- `ConformalConfig` in `src/sp500_forecast/config.py`
- `conformal:` block in `configs/sp500.yaml`
- ACI post-processing calls in `src/sp500_forecast/pipeline.py`
- `tests/test_conformal.py`

### Logic

For each test day:

1. Estimate causal rolling volatility using only data before the prediction day.
2. Compute historical standardized errors.
3. Take conformal quantile `q`.
4. Build `LowerBound`, `UpperBound`, and target-specific `HighBound` or `LowBound`.
5. After observing the actual value, update ACI alpha online.

### Verification

The conformal unit tests check:

- volatility uses only previous values;
- `HighBound` is produced;
- standardized error equals `|Actual - Predicted| / Volatility`.

## 2. Replace ATR Feature With RollingVolatility20

### Problem

ATR is a finance-specific technical indicator and may be harder to explain to the teacher. The user wanted a more intuitive explanation aligned with ACI.

### Reasoning

ATR was not needed for the ACI boundary. A simpler feature is:

```text
RollingVolatility20 = rolling std of close returns over 20 days
```

This is easier to explain:

> recent fluctuation of the price series

It avoids over-claiming finance-indicator expertise and is consistent with the ACI story: normalize errors by recent variability.

### Code Changes

In `src/sp500_forecast/features.py`:

```text
ATRPct -> RollingVolatility20
```

Removed the `_atr(...)` helper.

Updated:

- `tests/test_features.py`
- `README.md`
- architecture documentation

### Caveat

`outputs/sp500/summary_low.json` was generated before this rename, so it still shows `ATRPct`. The current code no longer produces `ATRPct`; rerun Low to refresh that output.

## 3. Synchronize Changes Into `/Users/francis/Desktop/ML_test`

### Problem

The first ACI implementation was made in `/Users/francis/Documents/New project 4`, but the user was running `/Users/francis/Desktop/ML_test`.

### Reasoning

The Desktop copy had additional feature-enhanced code that the root copy did not have. Directly overwriting files would have lost those changes.

### Code Changes

Selective synchronization into Desktop:

- added `conformal.py`;
- added `ConformalConfig`;
- patched `pipeline.py` to call ACI;
- patched `features.py` to use `RollingVolatility20`;
- updated config, README, and tests.

### Verification

Desktop tests passed:

```text
10 passed
```

## 4. Diagnose The First High Plot

### Observation

The first High plot after running the model showed the orange `Predicted` line systematically below the blue `Actual` line.

### Important Conclusion

This was not an ACI problem.

ACI only adds bounds; it does not change the point prediction:

```text
Predicted stays unchanged.
HighBound is added above Predicted.
```

The diagnostic numbers showed:

```text
model MAE  > naive MAE
model RMSE > naive RMSE
mean error was strongly negative
most days were under-predicted
```

### Reasoning

The point model was biased downward. ACI can cover the actual value with a wider upper bound, but it cannot make the center prediction better.

### Outcome

We kept ACI as the boundary layer and treated point-prediction bias as a separate modeling issue.

## 5. Change The Plot To Emphasize HighBound Instead Of Full Symmetric Interval

### Problem

For the High target, the full symmetric `LowerBound` / `UpperBound` shaded interval made the plot look large and visually noisy. The actual research question was about an upper boundary:

```text
Can HighBound cover Actual High?
```

### Reasoning

For High:

```text
Relevant side = Predicted -> HighBound
```

For Low:

```text
Relevant side = LowBound -> Predicted
```

### Code Changes

In `_save_fit_plot(...)`:

- High plots now shade only the upper ACI band;
- Low plots shade only the lower ACI band;
- fallback still supports full interval if target-specific columns are missing.

### Result

The plot legend changed from:

```text
ACI interval
```

to:

```text
ACI upper band
HighBound
```

This is why the later plot was definitely produced by the modified code.

## 6. Add Rolling Calibration Window For q

### Problem

Using all historical standardized errors to compute `q` made the interval slow to adapt:

- after extreme regimes, later intervals stayed too wide;
- before the regime was recognized, coverage could lag.

### Reasoning

The user’s original concern was that boundaries should not be dragged by extremes. That applies not only to raw errors, but also to old standardized errors.

So `q` should be computed from recent calibration scores:

```text
recent_scores = last calibration_window standardized errors
q_t = quantile(recent_scores)
```

### Code Changes

Added:

```yaml
conformal.calibration_window: 63
```

`63` means about one trading quarter.

### Logic

The code now keeps a rolling score history:

```text
score_t = |Actual_t - Predicted_t| / Volatility_t
q_t = quantile(score_{t-63:t-1})
```

### Why 63

It is a compromise:

- shorter than 252, so old crisis periods do not dominate forever;
- long enough to have enough calibration samples;
- easy to explain as approximately one quarter of trading days.

### Observed Effect On High

After recomputing ACI on existing High predictions:

```text
HighBound coverage ≈ 91.67%
ACI interval coverage ≈ 91.50%
mean interval width ≈ 239.63
```

The previous full-history version had wider average intervals.

## 7. Add Model Checkpoint Save / Load Switch

### Problem

Training the component neural models is expensive. The user asked for:

- a switch to retrain and save model weights;
- a switch to load existing saved weights and skip retraining.

### Reasoning

Saving only PyTorch weights is not enough because predictions depend on preprocessing:

- target `MinMaxScaler1D`;
- optional feature `StandardScaler2D`;
- model architecture;
- selected training params.

If these are not saved with the model, loading weights later can produce wrong predictions or shape mismatches.

### Code Changes

Added config:

```yaml
model.retrain_model: true
model.checkpoint_dir: "outputs/sp500/model_weights"
```

Added CLI flags:

```bash
--load-weights
--checkpoint-dir
```

Added training helpers:

```text
save_component_forecaster(...)
load_component_forecaster(...)
```

Added pipeline checkpoint helpers:

```text
_checkpoint_dir(...)
_component_checkpoint_path(...)
_load_forecasters_for_components(...)
_save_forecaster_checkpoints(...)
```

### Logic

When `model.retrain_model: true`:

1. Train every component model.
2. Save one checkpoint per component.
3. Record checkpoint paths in summary/component records.

When `model.retrain_model: false` or `--load-weights`:

1. Skip neural training.
2. Load component `.pt` files.
3. Reuse saved scalers and architecture metadata.
4. Predict validation and test windows.

### Checkpoint Layout

For High:

```text
outputs/sp500/model_weights/high/VIMF1.pt
outputs/sp500/model_weights/high/VIMF2.pt
outputs/sp500/model_weights/high/VIMF3.pt
outputs/sp500/model_weights/high/IMF2.pt
...
outputs/sp500/model_weights/high/Res.pt
```

For Low:

```text
outputs/sp500/model_weights/low/*.pt
```

### Checkpoint Contents

Each `.pt` contains:

- PyTorch `state_dict`;
- target scaler;
- feature scaler, if used;
- model architecture metadata;
- selected hyperparameters;
- component name;
- validation loss;
- epochs run.

### Verification

Added a round-trip checkpoint test:

1. Build a small `BiLSTMSAMTCN`.
2. Save it as a `ComponentForecaster`.
3. Load it back.
4. Check that predictions match.

Full Desktop tests:

```text
11 passed
```

Smoke verification:

1. Train smoke model.
2. Confirm `.pt` files are created.
3. Run again with `retrain_model=False`.
4. Confirm summary source is `model_checkpoints`.

## 8. Create Project Architecture Documentation

### Problem

The project had become difficult to explain:

- multiple decomposition models;
- multiple neural variants;
- High/Low each has many component models;
- feature counts differ for `Res` and non-`Res`;
- ACI is post-processing, not part of the neural model.

### Code / Documentation Changes

Created:

```text
reports/project_architecture.md
```

It documents:

- whole project flow;
- Mermaid architecture diagrams;
- model variants;
- decomposition modes;
- component model counts;
- feature counts per component;
- checkpoint flow;
- ACI formulas.

### Key Architecture Summary

Current component layout:

```text
VIMF1, VIMF2, VIMF3, IMF2, IMF3, IMF4, IMF5, IMF6, IMF7, Res
```

For each target:

```text
10 component models
```

For High + Low:

```text
20 component neural models
```

Feature count:

```text
non-Res component: 1 component value + 19 exogenous features = 20 features per timestep
Res component: 1 residual value + 0 exogenous features = 1 feature per timestep
```

With `window_size=3`:

```text
non-Res input shape: 3 x 20
Res input shape: 3 x 1
```

## 9. Important Current Caveats

### Low Output Is Stale

Current source code uses:

```text
RollingVolatility20
```

But the existing `outputs/sp500/summary_low.json` still shows:

```text
ATRPct
```

Reason: Low was generated before the ATR rename. Rerun Low to refresh.

### Formal High/Low Checkpoints Require Rerun

The checkpoint code exists now, but formal High/Low checkpoints are only created after rerunning:

```bash
python -m sp500_forecast.cli --config configs/sp500.yaml run --target high
python -m sp500_forecast.cli --config configs/sp500.yaml run --target low
```

After that, loading weights works with:

```bash
python -m sp500_forecast.cli --config configs/sp500.yaml run --target high --load-weights
```

### ACI Does Not Fix Point Prediction Bias

The orange `Predicted` line may still be biased. ACI only creates calibrated bounds:

```text
Predicted: point forecast
HighBound / LowBound: conformal boundary
```

If the goal is to improve the orange line, that is a separate model calibration or bias-correction task.

## 10. Short Version For Report Writing

The final method is:

1. Decompose High/Low into components using ICEEMDAN and PSO-VMD.
2. Train one BiLSTM-SAM-TCN model per component.
3. Concatenate 19 causal exogenous features to every non-residual component window.
4. Keep residual component model feature-free for stability.
5. In the current default, train and forecast component changes rather than component price levels.
6. Sum component predictions into `PredictedDelta`, then reconstruct price with `PredictedPrice_t = NaivePreviousValue_t + PredictedDelta_t`.
7. Apply ACI with volatility-standardized residuals to obtain HighBound / LowBound.
8. Save/load component model checkpoints to make experiments repeatable.

## 11. Delta-Target Revision

### Problem

The full S&P 500 paper-window experiment showed systematic point-forecast bias:

```text
original 2011-2023 run: train level below test level -> model underpredicted
downtrend probe: train level above test level -> model overpredicted
```

This indicated that direct price-level modeling was the main problem: the component models learned the training price level and had weak extrapolation when the test regime moved above or below the fit regime.

### Decision

Keep the reference paper's architecture but change the target definition:

```text
Old:
Price -> ICEEMDAN/VMD -> component level -> predict component level -> sum to price

New default:
Price change -> ICEEMDAN/VMD -> component changes -> predict next change -> restore price

PredictedPrice_t = NaivePreviousValue_t + ModelPredictedDelta_t
```

This preserves:

- ICEEMDAN first decomposition;
- VMD re-decomposition of the first IMF;
- one neural model per VIMF/IMF/Res component;
- BiLSTM -> SAM attention -> TCN -> dense head;
- walk-forward no-leakage decomposition;
- one-sided ACI boundary layer.

It changes only the modeling target from absolute price level to one-step price change.

### Code Changes

Added:

```text
experiment.target_transform: "delta"
```

Supported values:

```text
delta: decompose/forecast one-step price changes, then restore price
level: old direct price-level decomposition/forecasting
```

Prediction CSVs in delta mode now include:

```text
PredictedDelta
Predicted = NaivePreviousValue + PredictedDelta
```

Summaries record:

```text
target_transform
model_training_target
prediction_reconstruction
```

Checkpoint metadata also records `target_transform`, and the default checkpoint directory moved to:

```text
outputs/sp500/model_weights_delta
```

so old level checkpoints are not accidentally reused.
