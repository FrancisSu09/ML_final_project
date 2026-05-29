from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import AppConfig, resolve_path
from .conformal import add_aci_intervals
from .data import load_price_data
from .decomposition import DecompositionResult, decompose_signal
from .features import build_feature_frame
from .metrics import modified_diebold_mariano, regression_metrics
from .training import (
    ComponentForecaster,
    ComponentPrediction,
    candidate_params,
    choose_device,
    forecast_component_recursive,
    load_component_forecaster,
    save_component_forecaster,
    train_component,
    train_component_forecaster,
)


@dataclass(slots=True)
class TargetRunResult:
    target: str
    prediction_path: Path
    summary_path: Path
    component_path: Path | None
    plot_path: Path | None
    params_path: Path | None
    component_series_path: Path | None
    validation_path: Path | None
    metrics: dict[str, float]


def run_pipeline(config: AppConfig, *, targets: list[str] | None = None) -> list[TargetRunResult]:
    frame = load_price_data(config)
    frame = apply_time_scale(frame, config.experiment.time_scale)
    feature_frame = build_feature_frame(frame, config.features)
    feature_values = feature_frame.to_numpy(dtype=float) if not feature_frame.empty else None
    feature_names = list(feature_frame.columns)
    selected_targets = normalize_targets(targets or config.experiment.targets)
    output_dir = resolve_path(config.outputs.directory, base=config.base_dir)
    if output_dir is None:
        raise ValueError("outputs.directory cannot be empty")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for target in selected_targets:
        results.append(
            _run_one_target(
                frame,
                target=target,
                config=config,
                output_dir=output_dir,
                feature_values=feature_values,
                feature_names=feature_names,
            )
        )
    return results


def normalize_targets(targets: list[str]) -> list[str]:
    normalized = []
    for target in targets:
        value = target.strip().lower()
        if value in {"high", "h", "daily highest price", "highest"}:
            normalized.append("High")
        elif value in {"low", "l", "daily lowest price", "lowest"}:
            normalized.append("Low")
        else:
            raise ValueError(f"Unsupported target {target!r}; use High, Low, or both.")
    return normalized


def apply_time_scale(frame: pd.DataFrame, time_scale: int) -> pd.DataFrame:
    """Apply the paper's T-day high/low aggregation for robustness tests."""
    if time_scale < 1:
        raise ValueError("experiment.time_scale must be >= 1")
    if time_scale == 1:
        return frame
    usable_rows = (len(frame) // time_scale) * time_scale
    if usable_rows == 0:
        raise ValueError("Not enough rows for requested experiment.time_scale")
    frame = frame.iloc[:usable_rows].copy()

    grouped = np.arange(len(frame)) // time_scale
    aggregations = {"Date": "last", "High": "max", "Low": "min"}
    if "Open" in frame.columns:
        aggregations["Open"] = "first"
    if "Close" in frame.columns:
        aggregations["Close"] = "last"
    if "Volume" in frame.columns:
        aggregations["Volume"] = "sum"
    scaled = frame.assign(_group=grouped).groupby("_group", as_index=False).agg(aggregations)
    columns = ["Date", "High", "Low"]
    columns.extend([name for name in ("Open", "Close", "Volume") if name in scaled.columns])
    return scaled[columns].reset_index(drop=True)


def split_index_from_date(dates: pd.Series, split_date: str | None) -> int | None:
    if not split_date:
        return None
    split_timestamp = pd.Timestamp(split_date)
    date_values = dates.to_numpy(dtype="datetime64[ns]")
    split_index = int(np.searchsorted(date_values, np.datetime64(split_timestamp)))
    if split_index <= 0 or split_index >= len(dates):
        raise ValueError(
            f"experiment.train_split_date={split_date!r} leaves no train or test rows"
        )
    return split_index


def _run_one_target(
    frame: pd.DataFrame,
    *,
    target: str,
    config: AppConfig,
    output_dir: Path,
    feature_values: np.ndarray | None,
    feature_names: list[str],
) -> TargetRunResult:
    price_signal = frame[target].to_numpy(dtype=float)
    model_signal = _target_model_signal(price_signal, config=config)
    dates = pd.to_datetime(frame["Date"]).reset_index(drop=True)
    device = choose_device(config.experiment.device)
    split_index = effective_split_index(
        dates,
        train_ratio=config.experiment.train_ratio,
        split_date=config.experiment.train_split_date,
    )

    stem = target.lower()
    if config.decomposition.scope == "full_sample":
        return _run_full_sample_decomposition(
            price_signal=price_signal,
            model_signal=model_signal,
            dates=dates,
            target=target,
            stem=stem,
            config=config,
            output_dir=output_dir,
            split_index=split_index,
            device=device,
            feature_values=feature_values,
            feature_names=feature_names,
        )
    if config.decomposition.scope == "train_only_recursive":
        return _run_train_only_recursive_decomposition(
            price_signal=price_signal,
            model_signal=model_signal,
            dates=dates,
            target=target,
            stem=stem,
            config=config,
            output_dir=output_dir,
            split_index=split_index,
            device=device,
            feature_values=feature_values,
            feature_names=feature_names,
        )
    if config.decomposition.scope == "walk_forward":
        return _run_walk_forward_decomposition(
            price_signal=price_signal,
            model_signal=model_signal,
            dates=dates,
            target=target,
            stem=stem,
            config=config,
            output_dir=output_dir,
            split_index=split_index,
            device=device,
            feature_values=feature_values,
            feature_names=feature_names,
        )
    raise ValueError(
        "Unsupported decomposition.scope. Use 'full_sample', 'train_only_recursive', or 'walk_forward'."
    )


def effective_split_index(
    dates: pd.Series,
    *,
    train_ratio: float,
    split_date: str | None,
) -> int:
    split_index = split_index_from_date(dates, split_date)
    if split_index is None:
        split_index = int(len(dates) * train_ratio)
    return int(np.clip(split_index, 1, len(dates) - 1))


def _target_transform(config: AppConfig) -> str:
    value = str(config.experiment.target_transform).strip().lower()
    aliases = {
        "price": "level",
        "price_level": "level",
        "level": "level",
        "change": "delta",
        "price_change": "delta",
        "return_delta": "delta",
        "delta": "delta",
    }
    if value not in aliases:
        raise ValueError(
            "experiment.target_transform must be 'level' or 'delta' "
            f"(got {config.experiment.target_transform!r})"
        )
    return aliases[value]


def _target_model_signal(price_signal: np.ndarray, *, config: AppConfig) -> np.ndarray:
    """Return the series that decomposition/component models should forecast."""
    values = np.asarray(price_signal, dtype=float)
    transform = _target_transform(config)
    if transform == "level":
        return values.copy()
    delta = np.zeros_like(values, dtype=float)
    delta[1:] = values[1:] - values[:-1]
    return delta


def _restore_price_predictions(
    model_predicted: np.ndarray,
    *,
    price_signal: np.ndarray,
    target_indices: np.ndarray,
    config: AppConfig,
) -> np.ndarray:
    """Map component-level model outputs back to the reported price forecast."""
    predicted = np.asarray(model_predicted, dtype=float)
    transform = _target_transform(config)
    if transform == "level":
        return predicted
    indices = np.asarray(target_indices, dtype=int)
    if np.any(indices <= 0):
        raise ValueError("delta target_transform requires target indices greater than zero")
    return np.asarray(price_signal, dtype=float)[indices - 1] + predicted


def _predicted_delta_column(
    model_predicted: np.ndarray,
    *,
    config: AppConfig,
) -> np.ndarray | None:
    if _target_transform(config) != "delta":
        return None
    return np.asarray(model_predicted, dtype=float)


def _target_transform_summary(config: AppConfig) -> dict:
    transform = _target_transform(config)
    if transform == "level":
        return {
            "target_transform": "level",
            "model_training_target": "price_level_component_values",
            "prediction_reconstruction": "PredictedPrice_t = sum(ModelPredictedComponentLevel_t)",
        }
    return {
        "target_transform": "delta",
        "model_training_target": "price_change_component_values",
        "prediction_reconstruction": (
            "PredictedPrice_t = NaivePreviousValue_t + ModelPredictedDelta_t"
        ),
    }


def _run_full_sample_decomposition(
    *,
    price_signal: np.ndarray,
    model_signal: np.ndarray,
    dates: pd.Series,
    target: str,
    stem: str,
    config: AppConfig,
    output_dir: Path,
    split_index: int,
    device,
    feature_values: np.ndarray | None,
    feature_names: list[str],
) -> TargetRunResult:
    decomposition = decompose_signal(model_signal, config.decomposition, target=target)
    params_path = _param_cache_path(config, output_dir=output_dir, stem=stem)
    cached_params = _load_cached_params(params_path, config=config)
    checkpoint_dir = _checkpoint_dir(config, output_dir=output_dir, stem=stem)
    if not config.model.retrain_model:
        raise ValueError(
            "model.retrain_model=false is supported for train_only_recursive and "
            "walk_forward scopes. Use one of those scopes to load saved model weights."
        )
    component_series_path = _save_component_series(
        decomposition,
        dates=dates,
        path=output_dir / f"decomposition_components_{stem}.csv",
    )
    component_predictions: list[ComponentPrediction] = []
    for component in decomposition.components:
        component_predictions.append(
            train_component(
                name=component.name,
                series=component.values,
                config=config,
                device=device,
                split_index=split_index,
                cached_params=cached_params.get(component.name) if cached_params else None,
                features=_features_for_component(component.name, feature_values, config=config),
            )
        )

    target_indices = component_predictions[0].target_indices
    for prediction in component_predictions[1:]:
        if not np.array_equal(target_indices, prediction.target_indices):
            raise RuntimeError("Component predictions are not aligned on the same target indices")

    model_predicted = np.sum([prediction.predicted for prediction in component_predictions], axis=0)
    predicted = _restore_price_predictions(
        model_predicted,
        price_signal=price_signal,
        target_indices=target_indices,
        config=config,
    )
    actual = price_signal[target_indices]
    naive = price_signal[target_indices - 1]
    metrics = regression_metrics(actual, predicted)
    naive_metrics = regression_metrics(actual, naive)
    mdm = {
        loss: asdict(modified_diebold_mariano(actual, predicted, naive, loss=loss))
        for loss in ("mape", "mae", "rmse")
    }

    prediction_path = output_dir / f"predictions_{stem}.csv"
    prediction_frame = pd.DataFrame(
        {
            "Date": dates.iloc[target_indices].dt.strftime("%Y-%m-%d").to_numpy(),
            "TargetIndex": target_indices,
            "Actual": actual,
            "Predicted": predicted,
            "NaivePreviousValue": naive,
            "Error": predicted - actual,
        }
    )
    predicted_delta = _predicted_delta_column(model_predicted, config=config)
    if predicted_delta is not None:
        prediction_frame["PredictedDelta"] = predicted_delta
    prediction_frame, conformal_summary = add_aci_intervals(
        prediction_frame,
        target=target,
        signal=price_signal,
        calibration_frame=None,
        config=config.conformal,
    )
    prediction_frame.to_csv(prediction_path, index=False)

    component_path = None
    if config.outputs.save_component_predictions:
        component_path = output_dir / f"component_predictions_{stem}.csv"
        component_frame = pd.DataFrame(
            {
                "Date": dates.iloc[target_indices].dt.strftime("%Y-%m-%d").to_numpy(),
                "TargetIndex": target_indices,
            }
        )
        for prediction in component_predictions:
            component_frame[f"{prediction.name}_actual"] = prediction.actual
            component_frame[f"{prediction.name}_predicted"] = prediction.predicted
        component_frame.to_csv(component_path, index=False)

    plot_path = None
    if config.outputs.save_plots:
        plot_path = output_dir / f"fit_{stem}.png"
        _save_fit_plot(prediction_frame, target=target, path=plot_path)

    component_records = [
        {
            "name": prediction.name,
            "best_val_loss": prediction.best_val_loss,
            "params": prediction.params,
            "epochs_run": prediction.epochs_run,
        }
        for prediction in component_predictions
    ]
    forecasters = [
        prediction.forecaster
        for prediction in component_predictions
        if prediction.forecaster is not None
    ]
    checkpoint_paths = _save_forecaster_checkpoints(
        forecasters,
        checkpoint_dir=checkpoint_dir,
        config=config,
        target=target,
    )
    _attach_checkpoint_paths(component_records, checkpoint_paths)

    summary = {
        "target": target,
        "rows": int(len(price_signal)),
        "test_rows": int(len(actual)),
        "paper_method": "ICEEMDAN-PSO-VMD-BiLSTM-SAM-TCN",
        "decomposition_mode": config.decomposition.mode,
        "decomposition_scope": config.decomposition.scope,
        **_target_transform_summary(config),
        "model_variant": config.model.variant,
        "features": _feature_summary(feature_values, feature_names, config=config),
        "time_scale": config.experiment.time_scale,
        "train_split_date": config.experiment.train_split_date,
        "train_split_index": split_index,
        "metrics": metrics.as_dict(),
        "conformal": conformal_summary.as_dict(),
        "model_checkpoints": _checkpoint_summary(config, checkpoint_dir),
        "naive_previous_value_metrics": naive_metrics.as_dict(),
        "mdm_vs_naive": mdm,
        "vmd": {
            "k": decomposition.vmd_k,
            "alpha": decomposition.vmd_alpha,
            "pso_fitness": decomposition.pso_fitness,
            "imf1_reconstruction_error": _vmd_reconstruction_error(decomposition),
        },
        "components": component_records,
    }
    summary_path = output_dir / f"summary_{stem}.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    _save_decomposition_summary(decomposition, output_dir / f"decomposition_{stem}.npz")
    if config.model.save_best_params:
        _save_best_params(
            params_path,
            target=target,
            decomposition=decomposition,
            config=config,
            component_records=component_records,
        )

    return TargetRunResult(
        target=target,
        prediction_path=prediction_path,
        summary_path=summary_path,
        component_path=component_path,
        plot_path=plot_path,
        params_path=params_path if config.model.save_best_params else None,
        component_series_path=component_series_path,
        validation_path=None,
        metrics=metrics.as_dict(),
    )


def _run_train_only_recursive_decomposition(
    *,
    price_signal: np.ndarray,
    model_signal: np.ndarray,
    dates: pd.Series,
    target: str,
    stem: str,
    config: AppConfig,
    output_dir: Path,
    split_index: int,
    device,
    feature_values: np.ndarray | None,
    feature_names: list[str],
) -> TargetRunResult:
    fit_end, validation_start, validation_end = fit_validation_test_boundaries(
        series_length=len(model_signal),
        window_size=config.experiment.window_size,
        split_index=split_index,
        validation_ratio=config.experiment.validation_ratio,
    )
    fit_signal = model_signal[:fit_end]
    decomposition = decompose_signal(fit_signal, config.decomposition, target=target)
    params_path = _param_cache_path(config, output_dir=output_dir, stem=stem)
    cached_params = _load_cached_params(params_path, config=config)
    checkpoint_dir = _checkpoint_dir(config, output_dir=output_dir, stem=stem)
    component_series_path = _save_component_series(
        decomposition,
        dates=dates.iloc[:fit_end].reset_index(drop=True),
        path=output_dir / f"decomposition_fit_components_{stem}.csv",
    )

    validation_steps = validation_end - validation_start
    forecast_steps = len(model_signal) - fit_end
    test_offset = split_index - fit_end
    target_indices = np.arange(split_index, len(model_signal), dtype=int)

    if not config.model.retrain_model:
        forecasters = _load_forecasters_for_components(
            decomposition,
            config=config,
            device=device,
            checkpoint_dir=checkpoint_dir,
        )
        selection = {
            "source": "model_checkpoints",
            "metric": "RMSE",
            "score": None,
            "validation_metrics": None,
            "checkpoint_dir": str(checkpoint_dir),
        }
    elif cached_params:
        forecasters = _train_forecasters_for_components(
            decomposition,
            config=config,
            device=device,
            params_by_component=cached_params,
            features=feature_values[:fit_end] if feature_values is not None else None,
        )
        selection = {
            "source": "cached_component_params",
            "metric": "RMSE",
            "score": None,
            "validation_metrics": None,
        }
    else:
        forecasters, selection = _select_forecasters_by_recursive_validation(
            price_signal=price_signal,
            decomposition=decomposition,
            validation_start=validation_start,
            validation_end=validation_end,
            config=config,
            device=device,
            features=feature_values[:fit_end] if feature_values is not None else None,
            validation_feature_windows=_feature_windows(
                feature_values,
                start=validation_start,
                end=validation_end,
                window_size=config.experiment.window_size,
            ),
        )

    all_component_predictions = _forecast_components_recursive(
        forecasters,
        decomposition,
        steps=forecast_steps,
        config=config,
        device=device,
        feature_windows=_feature_windows(
            feature_values,
            start=fit_end,
            end=fit_end + forecast_steps,
            window_size=config.experiment.window_size,
        ),
    )
    validation_component_predictions = {
        name: values[:validation_steps]
        for name, values in all_component_predictions.items()
    }
    component_predictions = {
        name: values[test_offset:]
        for name, values in all_component_predictions.items()
    }

    validation_model_predicted = np.sum(list(validation_component_predictions.values()), axis=0)
    validation_indices = np.arange(validation_start, validation_end, dtype=int)
    validation_predicted = _restore_price_predictions(
        validation_model_predicted,
        price_signal=price_signal,
        target_indices=validation_indices,
        config=config,
    )
    validation_actual = price_signal[validation_start:validation_end]
    validation_naive = price_signal[validation_start - 1 : validation_end - 1]
    validation_metrics = regression_metrics(validation_actual, validation_predicted)
    if selection.get("source") == "model_checkpoints":
        selection["score"] = validation_metrics.rmse
        selection["validation_metrics"] = validation_metrics.as_dict()

    model_predicted = np.sum(list(component_predictions.values()), axis=0)
    predicted = _restore_price_predictions(
        model_predicted,
        price_signal=price_signal,
        target_indices=target_indices,
        config=config,
    )
    actual = price_signal[target_indices]
    naive = price_signal[target_indices - 1]
    metrics = regression_metrics(actual, predicted)
    naive_metrics = regression_metrics(actual, naive)
    mdm = {
        loss: asdict(modified_diebold_mariano(actual, predicted, naive, loss=loss))
        for loss in ("mape", "mae", "rmse")
    }

    prediction_path = output_dir / f"predictions_{stem}.csv"
    prediction_frame = pd.DataFrame(
        {
            "Date": dates.iloc[target_indices].dt.strftime("%Y-%m-%d").to_numpy(),
            "TargetIndex": target_indices,
            "Actual": actual,
            "Predicted": predicted,
            "NaivePreviousValue": naive,
            "Error": predicted - actual,
        }
    )
    predicted_delta = _predicted_delta_column(model_predicted, config=config)
    if predicted_delta is not None:
        prediction_frame["PredictedDelta"] = predicted_delta
    validation_path = output_dir / f"validation_predictions_{stem}.csv"
    validation_frame = pd.DataFrame(
        {
            "Date": dates.iloc[validation_start:validation_end].dt.strftime("%Y-%m-%d").to_numpy(),
            "TargetIndex": validation_indices,
            "Actual": validation_actual,
            "Predicted": validation_predicted,
            "NaivePreviousValue": validation_naive,
            "Error": validation_predicted - validation_actual,
        }
    )
    validation_predicted_delta = _predicted_delta_column(validation_model_predicted, config=config)
    if validation_predicted_delta is not None:
        validation_frame["PredictedDelta"] = validation_predicted_delta
    validation_frame.to_csv(validation_path, index=False)
    prediction_frame, conformal_summary = add_aci_intervals(
        prediction_frame,
        target=target,
        signal=price_signal,
        calibration_frame=validation_frame,
        config=config.conformal,
    )
    prediction_frame.to_csv(prediction_path, index=False)

    component_path = None
    if config.outputs.save_component_predictions:
        component_path = output_dir / f"component_predictions_{stem}.csv"
        component_frame = pd.DataFrame(
            {
                "Date": dates.iloc[target_indices].dt.strftime("%Y-%m-%d").to_numpy(),
                "TargetIndex": target_indices,
            }
        )
        for name, values in component_predictions.items():
            component_frame[f"{name}_predicted"] = values
        component_frame.to_csv(component_path, index=False)

    plot_path = None
    if config.outputs.save_plots:
        plot_path = output_dir / f"fit_{stem}.png"
        _save_fit_plot(prediction_frame, target=target, path=plot_path)

    component_records = [
        {
            "name": forecaster.name,
            "best_val_loss": forecaster.best_val_loss,
            "params": forecaster.params,
            "epochs_run": forecaster.epochs_run,
        }
        for forecaster in forecasters
    ]
    checkpoint_paths = (
        _save_forecaster_checkpoints(
            forecasters,
            checkpoint_dir=checkpoint_dir,
            config=config,
            target=target,
        )
        if config.model.retrain_model
        else {forecaster.name: forecaster.checkpoint_path for forecaster in forecasters}
    )
    _attach_checkpoint_paths(component_records, checkpoint_paths)
    summary = {
        "target": target,
        "rows": int(len(price_signal)),
        "test_rows": int(len(actual)),
        "paper_method": "ICEEMDAN-PSO-VMD-BiLSTM-SAM-TCN",
        "decomposition_mode": config.decomposition.mode,
        "decomposition_scope": config.decomposition.scope,
        **_target_transform_summary(config),
        "model_variant": config.model.variant,
        "features": _feature_summary(feature_values, feature_names, config=config),
        "time_scale": config.experiment.time_scale,
        "train_split_date": config.experiment.train_split_date,
        "train_split_index": split_index,
        "fit_end_index": fit_end,
        "validation_start_index": validation_start,
        "validation_end_index": validation_end,
        "validation_rows": int(len(validation_actual)),
        "validation_metrics": validation_metrics.as_dict(),
        "parameter_selection": selection,
        "metrics": metrics.as_dict(),
        "conformal": conformal_summary.as_dict(),
        "model_checkpoints": _checkpoint_summary(config, checkpoint_dir),
        "naive_previous_value_metrics": naive_metrics.as_dict(),
        "mdm_vs_naive": mdm,
        "vmd": {
            "k": decomposition.vmd_k,
            "alpha": decomposition.vmd_alpha,
            "pso_fitness": decomposition.pso_fitness,
            "imf1_reconstruction_error": _vmd_reconstruction_error(decomposition),
        },
        "components": component_records,
    }
    summary_path = output_dir / f"summary_{stem}.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    _save_decomposition_summary(decomposition, output_dir / f"decomposition_fit_{stem}.npz")
    if config.model.save_best_params:
        _save_best_params(
            params_path,
            target=target,
            decomposition=decomposition,
            config=config,
            component_records=component_records,
            selection=selection,
            split={
                "fit_end_index": fit_end,
                "validation_start_index": validation_start,
                "validation_end_index": validation_end,
                "train_split_index": split_index,
            },
        )

    return TargetRunResult(
        target=target,
        prediction_path=prediction_path,
        summary_path=summary_path,
        component_path=component_path,
        plot_path=plot_path,
        params_path=params_path if config.model.save_best_params else None,
        component_series_path=component_series_path,
        validation_path=validation_path,
        metrics=metrics.as_dict(),
    )


def _run_walk_forward_decomposition(
    *,
    price_signal: np.ndarray,
    model_signal: np.ndarray,
    dates: pd.Series,
    target: str,
    stem: str,
    config: AppConfig,
    output_dir: Path,
    split_index: int,
    device,
    feature_values: np.ndarray | None,
    feature_names: list[str],
) -> TargetRunResult:
    fit_end, validation_start, validation_end = fit_validation_test_boundaries(
        series_length=len(model_signal),
        window_size=config.experiment.window_size,
        split_index=split_index,
        validation_ratio=config.experiment.validation_ratio,
    )
    fit_signal = model_signal[:fit_end]
    fit_decomposition = decompose_signal(fit_signal, config.decomposition, target=target)
    params_path = _param_cache_path(config, output_dir=output_dir, stem=stem)
    cached_params = _load_cached_params(params_path, config=config)
    checkpoint_dir = _checkpoint_dir(config, output_dir=output_dir, stem=stem)
    component_series_path = _save_component_series(
        fit_decomposition,
        dates=dates.iloc[:fit_end].reset_index(drop=True),
        path=output_dir / f"decomposition_fit_components_{stem}.csv",
    )
    component_names = [component.name for component in fit_decomposition.components]

    validation_windows, validation_feature_windows, validation_window_diagnostics = _precompute_walk_forward_windows(
        signal=model_signal,
        start=validation_start,
        end=validation_end,
        component_names=component_names,
        config=config,
        target=target,
        feature_values=feature_values,
    )
    test_windows, test_feature_windows, test_window_diagnostics = _precompute_walk_forward_windows(
        signal=model_signal,
        start=split_index,
        end=len(model_signal),
        component_names=component_names,
        config=config,
        target=target,
        feature_values=feature_values,
    )

    if not config.model.retrain_model:
        forecasters = _load_forecasters_for_components(
            fit_decomposition,
            config=config,
            device=device,
            checkpoint_dir=checkpoint_dir,
        )
        validation_component_predictions = _predict_component_windows(
            forecasters,
            validation_windows,
            config=config,
            device=device,
            feature_windows=validation_feature_windows,
        )
        validation_model_predicted = np.sum(list(validation_component_predictions.values()), axis=0)
        validation_indices = np.arange(validation_start, validation_end, dtype=int)
        validation_predicted = _restore_price_predictions(
            validation_model_predicted,
            price_signal=price_signal,
            target_indices=validation_indices,
            config=config,
        )
        validation_actual = price_signal[validation_start:validation_end]
        validation_metrics = regression_metrics(validation_actual, validation_predicted)
        selection = {
            "source": "model_checkpoints",
            "metric": "RMSE",
            "score": validation_metrics.rmse,
            "validation_metrics": validation_metrics.as_dict(),
            "candidates_evaluated": 0,
            "checkpoint_dir": str(checkpoint_dir),
        }
    elif cached_params:
        forecasters = _train_forecasters_for_components(
            fit_decomposition,
            config=config,
            device=device,
            params_by_component=cached_params,
            features=feature_values[:fit_end] if feature_values is not None else None,
        )
        validation_component_predictions = _predict_component_windows(
            forecasters,
            validation_windows,
            config=config,
            device=device,
            feature_windows=validation_feature_windows,
        )
        validation_model_predicted = np.sum(list(validation_component_predictions.values()), axis=0)
        validation_indices = np.arange(validation_start, validation_end, dtype=int)
        validation_predicted = _restore_price_predictions(
            validation_model_predicted,
            price_signal=price_signal,
            target_indices=validation_indices,
            config=config,
        )
        validation_actual = price_signal[validation_start:validation_end]
        validation_metrics = regression_metrics(validation_actual, validation_predicted)
        selection = {
            "source": "cached_component_params",
            "metric": "RMSE",
            "score": validation_metrics.rmse,
            "validation_metrics": validation_metrics.as_dict(),
            "candidates_evaluated": 0,
        }
    else:
        forecasters, selection, validation_component_predictions = (
            _select_forecasters_by_walk_forward_validation(
                price_signal=price_signal,
                fit_decomposition=fit_decomposition,
                validation_start=validation_start,
                validation_end=validation_end,
                validation_windows=validation_windows,
                validation_feature_windows=validation_feature_windows,
                config=config,
                device=device,
                features=feature_values[:fit_end] if feature_values is not None else None,
            )
        )
        validation_model_predicted = np.sum(list(validation_component_predictions.values()), axis=0)
        validation_indices = np.arange(validation_start, validation_end, dtype=int)
        validation_predicted = _restore_price_predictions(
            validation_model_predicted,
            price_signal=price_signal,
            target_indices=validation_indices,
            config=config,
        )
        validation_actual = price_signal[validation_start:validation_end]
        validation_metrics = regression_metrics(validation_actual, validation_predicted)

    test_component_predictions = _predict_component_windows(
        forecasters,
        test_windows,
        config=config,
        device=device,
        feature_windows=test_feature_windows,
    )
    model_predicted = np.sum(list(test_component_predictions.values()), axis=0)
    target_indices = np.arange(split_index, len(model_signal), dtype=int)
    predicted = _restore_price_predictions(
        model_predicted,
        price_signal=price_signal,
        target_indices=target_indices,
        config=config,
    )
    actual = price_signal[target_indices]
    naive = price_signal[target_indices - 1]
    metrics = regression_metrics(actual, predicted)
    naive_metrics = regression_metrics(actual, naive)
    mdm = {
        loss: asdict(modified_diebold_mariano(actual, predicted, naive, loss=loss))
        for loss in ("mape", "mae", "rmse")
    }

    prediction_path = output_dir / f"predictions_{stem}.csv"
    prediction_frame = pd.DataFrame(
        {
            "Date": dates.iloc[target_indices].dt.strftime("%Y-%m-%d").to_numpy(),
            "TargetIndex": target_indices,
            "Actual": actual,
            "Predicted": predicted,
            "NaivePreviousValue": naive,
            "Error": predicted - actual,
        }
    )
    predicted_delta = _predicted_delta_column(model_predicted, config=config)
    if predicted_delta is not None:
        prediction_frame["PredictedDelta"] = predicted_delta
    validation_path = output_dir / f"validation_predictions_{stem}.csv"
    validation_naive = price_signal[validation_start - 1 : validation_end - 1]
    validation_frame = pd.DataFrame(
        {
            "Date": dates.iloc[validation_start:validation_end].dt.strftime("%Y-%m-%d").to_numpy(),
            "TargetIndex": validation_indices,
            "Actual": validation_actual,
            "Predicted": validation_predicted,
            "NaivePreviousValue": validation_naive,
            "Error": validation_predicted - validation_actual,
        }
    )
    validation_predicted_delta = _predicted_delta_column(validation_model_predicted, config=config)
    if validation_predicted_delta is not None:
        validation_frame["PredictedDelta"] = validation_predicted_delta
    validation_frame.to_csv(validation_path, index=False)
    prediction_frame, conformal_summary = add_aci_intervals(
        prediction_frame,
        target=target,
        signal=price_signal,
        calibration_frame=validation_frame,
        config=config.conformal,
    )
    prediction_frame.to_csv(prediction_path, index=False)

    component_path = None
    if config.outputs.save_component_predictions:
        component_path = output_dir / f"component_predictions_{stem}.csv"
        component_frame = pd.DataFrame(
            {
                "Date": dates.iloc[target_indices].dt.strftime("%Y-%m-%d").to_numpy(),
                "TargetIndex": target_indices,
            }
        )
        for name, values in test_component_predictions.items():
            component_frame[f"{name}_predicted"] = values
        component_frame.to_csv(component_path, index=False)

    plot_path = None
    if config.outputs.save_plots:
        plot_path = output_dir / f"fit_{stem}.png"
        _save_fit_plot(prediction_frame, target=target, path=plot_path)

    component_records = [
        {
            "name": forecaster.name,
            "best_val_loss": forecaster.best_val_loss,
            "params": forecaster.params,
            "epochs_run": forecaster.epochs_run,
        }
        for forecaster in forecasters
    ]
    checkpoint_paths = (
        _save_forecaster_checkpoints(
            forecasters,
            checkpoint_dir=checkpoint_dir,
            config=config,
            target=target,
        )
        if config.model.retrain_model
        else {forecaster.name: forecaster.checkpoint_path for forecaster in forecasters}
    )
    _attach_checkpoint_paths(component_records, checkpoint_paths)
    summary = {
        "target": target,
        "rows": int(len(price_signal)),
        "test_rows": int(len(actual)),
        "paper_method": "ICEEMDAN-PSO-VMD-BiLSTM-SAM-TCN",
        "decomposition_mode": config.decomposition.mode,
        "decomposition_scope": config.decomposition.scope,
        **_target_transform_summary(config),
        "model_variant": config.model.variant,
        "features": _feature_summary(feature_values, feature_names, config=config),
        "time_scale": config.experiment.time_scale,
        "train_split_date": config.experiment.train_split_date,
        "train_split_index": split_index,
        "fit_end_index": fit_end,
        "validation_start_index": validation_start,
        "validation_end_index": validation_end,
        "validation_rows": int(len(validation_actual)),
        "validation_metrics": validation_metrics.as_dict(),
        "parameter_selection": selection,
        "walk_forward": {
            "history_policy": "actual_values_through_previous_day",
            "validation_window_diagnostics": validation_window_diagnostics,
            "test_window_diagnostics": test_window_diagnostics,
        },
        "metrics": metrics.as_dict(),
        "conformal": conformal_summary.as_dict(),
        "model_checkpoints": _checkpoint_summary(config, checkpoint_dir),
        "naive_previous_value_metrics": naive_metrics.as_dict(),
        "mdm_vs_naive": mdm,
        "vmd": {
            "k": fit_decomposition.vmd_k,
            "alpha": fit_decomposition.vmd_alpha,
            "pso_fitness": fit_decomposition.pso_fitness,
            "imf1_reconstruction_error": _vmd_reconstruction_error(fit_decomposition),
        },
        "components": component_records,
    }
    summary_path = output_dir / f"summary_{stem}.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    _save_decomposition_summary(fit_decomposition, output_dir / f"decomposition_fit_{stem}.npz")
    if config.model.save_best_params:
        _save_best_params(
            params_path,
            target=target,
            decomposition=fit_decomposition,
            config=config,
            component_records=component_records,
            selection=selection,
            split={
                "fit_end_index": fit_end,
                "validation_start_index": validation_start,
                "validation_end_index": validation_end,
                "train_split_index": split_index,
            },
        )

    return TargetRunResult(
        target=target,
        prediction_path=prediction_path,
        summary_path=summary_path,
        component_path=component_path,
        plot_path=plot_path,
        params_path=params_path if config.model.save_best_params else None,
        component_series_path=component_series_path,
        validation_path=validation_path,
        metrics=metrics.as_dict(),
    )


def fit_validation_test_boundaries(
    *,
    series_length: int,
    window_size: int,
    split_index: int,
    validation_ratio: float,
) -> tuple[int, int, int]:
    pre_test_targets = np.arange(window_size, split_index, dtype=int)
    if len(pre_test_targets) == 0:
        raise ValueError("No pre-test windows. Increase data length or lower window_size.")
    val_count = max(1, int(len(pre_test_targets) * validation_ratio))
    if val_count >= len(pre_test_targets):
        val_count = 1
    fit_targets = pre_test_targets[:-val_count]
    if len(fit_targets) == 0:
        raise ValueError("No fit windows after validation split.")
    fit_end = int(np.max(fit_targets)) + 1
    validation_start = fit_end
    validation_end = int(np.clip(split_index, validation_start + 1, series_length))
    return fit_end, validation_start, validation_end


def _select_forecasters_by_recursive_validation(
    *,
    price_signal: np.ndarray,
    decomposition: DecompositionResult,
    validation_start: int,
    validation_end: int,
    config: AppConfig,
    device,
    features: np.ndarray | None = None,
    validation_feature_windows: np.ndarray | None = None,
) -> tuple[list[ComponentForecaster], dict]:
    candidates = candidate_params(config)
    validation_steps = validation_end - validation_start
    if validation_steps <= 0:
        raise ValueError("Validation horizon must contain at least one row.")

    best_forecasters: list[ComponentForecaster] | None = None
    best_metrics = None
    best_params: dict[str, int | float] | None = None
    best_score = float("inf")
    for params in candidates:
        forecasters = _train_forecasters_for_components(
            decomposition,
            config=config,
            device=device,
            shared_params=params,
            features=features,
        )
        component_predictions = _forecast_components_recursive(
            forecasters,
            decomposition,
            steps=validation_steps,
            config=config,
            device=device,
            feature_windows=validation_feature_windows,
        )
        model_predicted = np.sum(list(component_predictions.values()), axis=0)
        validation_indices = np.arange(validation_start, validation_end, dtype=int)
        predicted = _restore_price_predictions(
            model_predicted,
            price_signal=price_signal,
            target_indices=validation_indices,
            config=config,
        )
        actual = price_signal[validation_start:validation_end]
        metrics = regression_metrics(actual, predicted)
        score = metrics.rmse
        if score < best_score:
            best_score = score
            best_forecasters = forecasters
            best_metrics = metrics
            best_params = params.copy()

    if best_forecasters is None or best_metrics is None or best_params is None:
        raise RuntimeError("Unable to select component forecasters by recursive validation")

    return best_forecasters, {
        "source": (
            "recursive_validation_grid_search"
            if config.model.search_hyperparameters
            else "default_params_recursive_validation"
        ),
        "metric": "RMSE",
        "score": best_score,
        "validation_metrics": best_metrics.as_dict(),
        "shared_params": best_params,
        "candidates_evaluated": len(candidates),
    }


def _train_forecasters_for_components(
    decomposition: DecompositionResult,
    *,
    config: AppConfig,
    device,
    shared_params: dict[str, int | float] | None = None,
    params_by_component: dict[str, dict[str, int | float]] | None = None,
    features: np.ndarray | None = None,
) -> list[ComponentForecaster]:
    forecasters: list[ComponentForecaster] = []
    for component in decomposition.components:
        params = shared_params
        if params_by_component is not None:
            params = params_by_component.get(component.name, shared_params)
        forecasters.append(
            train_component_forecaster(
                name=component.name,
                series=component.values,
                config=config,
                device=device,
                cached_params=params,
                features=_features_for_component(component.name, features, config=config),
            )
        )
    return forecasters


def _forecast_components_recursive(
    forecasters: list[ComponentForecaster],
    decomposition: DecompositionResult,
    *,
    steps: int,
    config: AppConfig,
    device,
    feature_windows: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    histories = {component.name: component.values for component in decomposition.components}
    return {
        forecaster.name: forecast_component_recursive(
            forecaster,
            histories[forecaster.name],
            steps=steps,
            window_size=config.experiment.window_size,
            device=device,
            feature_windows=feature_windows,
        )
        for forecaster in forecasters
    }


def _precompute_walk_forward_windows(
    *,
    signal: np.ndarray,
    start: int,
    end: int,
    component_names: list[str],
    config: AppConfig,
    target: str,
    feature_values: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray | None, dict]:
    steps = max(0, end - start)
    window_size = config.experiment.window_size
    windows = {
        name: np.zeros((steps, window_size), dtype=float)
        for name in component_names
    }
    feature_windows = _feature_windows(
        feature_values,
        start=start,
        end=end,
        window_size=window_size,
    )
    missing_counts = {name: 0 for name in component_names}
    short_counts = {name: 0 for name in component_names}
    extra_component_counts: dict[str, int] = {}
    component_name_set = set(component_names)

    for offset, target_index in enumerate(range(start, end)):
        history = signal[:target_index]
        decomposition = decompose_signal(history, config.decomposition, target=target)
        available, extra_names, folded_target = _align_walk_forward_components(
            decomposition,
            component_names=component_names,
        )
        for extra_name in extra_names:
            extra_component_counts[extra_name] = extra_component_counts.get(extra_name, 0) + 1
        for name in component_names:
            values = available.get(name)
            if values is None:
                missing_counts[name] += 1
                continue
            if len(values) < window_size:
                short_counts[name] += 1
                windows[name][offset, -len(values) :] = values
                continue
            windows[name][offset] = values[-window_size:]

    diagnostics = {
        "steps": steps,
        "start_index": int(start),
        "end_index": int(end),
        "missing_component_counts": missing_counts,
        "short_component_counts": short_counts,
        "extra_component_counts": extra_component_counts,
        "extra_component_policy": (
            f"fold_into_{folded_target}" if folded_target else "drop_when_no_component_exists"
        ),
    }
    return windows, feature_windows, diagnostics


def _align_walk_forward_components(
    decomposition: DecompositionResult,
    *,
    component_names: list[str],
) -> tuple[dict[str, np.ndarray], list[str], str | None]:
    """Align each walk-forward decomposition to the fit-time component layout."""
    component_name_set = set(component_names)
    available = {
        component.name: np.asarray(component.values, dtype=float).copy()
        for component in decomposition.components
        if component.name in component_name_set
    }
    extras = [
        component
        for component in decomposition.components
        if component.name not in component_name_set
    ]
    folded_target = _extra_component_fold_target(component_names)
    if extras and folded_target:
        template = extras[0].values
        folded_values = available.get(
            folded_target,
            np.zeros_like(np.asarray(template, dtype=float)),
        )
        for component in extras:
            folded_values = folded_values + np.asarray(component.values, dtype=float)
        available[folded_target] = folded_values
    return available, [component.name for component in extras], folded_target


def _extra_component_fold_target(component_names: list[str]) -> str | None:
    if not component_names:
        return None
    if "Res" in component_names:
        return "Res"
    return component_names[-1]


def _select_forecasters_by_walk_forward_validation(
    *,
    price_signal: np.ndarray,
    fit_decomposition: DecompositionResult,
    validation_start: int,
    validation_end: int,
    validation_windows: dict[str, np.ndarray],
    validation_feature_windows: np.ndarray | None,
    config: AppConfig,
    device,
    features: np.ndarray | None = None,
) -> tuple[list[ComponentForecaster], dict, dict[str, np.ndarray]]:
    candidates = candidate_params(config)
    validation_actual = price_signal[validation_start:validation_end]
    if len(validation_actual) == 0:
        raise ValueError("Validation horizon must contain at least one row.")

    best_forecasters: list[ComponentForecaster] | None = None
    best_component_predictions: dict[str, np.ndarray] | None = None
    best_metrics = None
    best_params: dict[str, int | float] | None = None
    best_score = float("inf")
    for params in candidates:
        forecasters = _train_forecasters_for_components(
            fit_decomposition,
            config=config,
            device=device,
            shared_params=params,
            features=features,
        )
        component_predictions = _predict_component_windows(
            forecasters,
            validation_windows,
            config=config,
            device=device,
            feature_windows=validation_feature_windows,
        )
        model_predicted = np.sum(list(component_predictions.values()), axis=0)
        validation_indices = np.arange(validation_start, validation_end, dtype=int)
        predicted = _restore_price_predictions(
            model_predicted,
            price_signal=price_signal,
            target_indices=validation_indices,
            config=config,
        )
        metrics = regression_metrics(validation_actual, predicted)
        score = metrics.rmse
        if score < best_score:
            best_score = score
            best_forecasters = forecasters
            best_component_predictions = component_predictions
            best_metrics = metrics
            best_params = params.copy()

    if (
        best_forecasters is None
        or best_component_predictions is None
        or best_metrics is None
        or best_params is None
    ):
        raise RuntimeError("Unable to select component forecasters by walk-forward validation")

    return best_forecasters, {
        "source": (
            "walk_forward_validation_grid_search"
            if config.model.search_hyperparameters
            else "default_params_walk_forward_validation"
        ),
        "metric": "RMSE",
        "score": best_score,
        "validation_metrics": best_metrics.as_dict(),
        "shared_params": best_params,
        "candidates_evaluated": len(candidates),
    }, best_component_predictions


def _predict_component_windows(
    forecasters: list[ComponentForecaster],
    windows_by_name: dict[str, np.ndarray],
    *,
    config: AppConfig,
    device,
    feature_windows: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    return {
        forecaster.name: _predict_windows(
            forecaster,
            windows_by_name[forecaster.name],
            batch_size=int(forecaster.params.get("batch_size", config.model.batch_size)),
            device=device,
            feature_windows=feature_windows,
        )
        for forecaster in forecasters
    }


def _predict_windows(
    forecaster: ComponentForecaster,
    windows: np.ndarray,
    *,
    batch_size: int,
    device,
    feature_windows: np.ndarray | None = None,
) -> np.ndarray:
    if len(windows) == 0:
        return np.array([], dtype=float)
    component_scaled = forecaster.scaler.transform(np.asarray(windows, dtype=float)).reshape(
        len(windows),
        windows.shape[1],
        1,
    )
    if feature_windows is not None and forecaster.feature_scaler is not None:
        raw_features = np.asarray(feature_windows, dtype=float)
        scaled_features = forecaster.feature_scaler.transform(
            raw_features.reshape(-1, raw_features.shape[-1])
        ).reshape(raw_features.shape)
        x_scaled = np.concatenate([component_scaled, scaled_features], axis=2)
    else:
        x_scaled = component_scaled
    forecaster.model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x_scaled), max(batch_size, 1)):
            batch = torch.as_tensor(
                x_scaled[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            pred = forecaster.model(batch).detach().cpu().numpy().reshape(-1)
            predictions.append(pred)
    predicted_scaled = np.concatenate(predictions) if predictions else np.array([], dtype=float)
    return forecaster.scaler.inverse_transform(predicted_scaled)


def _feature_windows(
    feature_values: np.ndarray | None,
    *,
    start: int,
    end: int,
    window_size: int,
) -> np.ndarray | None:
    if feature_values is None:
        return None
    values = np.asarray(feature_values, dtype=float)
    steps = max(0, end - start)
    windows = np.zeros((steps, window_size, values.shape[1]), dtype=float)
    for offset, target_index in enumerate(range(start, end)):
        begin = max(0, target_index - window_size)
        window = values[begin:target_index]
        if len(window) == 0:
            continue
        windows[offset, -len(window) :] = window
    return windows


def _feature_summary(
    feature_values: np.ndarray | None,
    feature_names: list[str],
    *,
    config: AppConfig,
) -> dict:
    return {
        "enabled": feature_values is not None and len(feature_names) > 0,
        "count": len(feature_names),
        "names": feature_names,
        "lag_policy": "feature_window_uses_target_index_minus_window_through_target_index_minus_1",
        "scaler": "per_component_standard_scaler_fit_on_component_fit_rows",
        "use_for_res": bool(config.features.use_for_res),
    }


def _features_for_component(
    component_name: str,
    feature_values: np.ndarray | None,
    *,
    config: AppConfig,
) -> np.ndarray | None:
    if feature_values is None:
        return None
    if component_name == "Res" and not config.features.use_for_res:
        return None
    return feature_values


def _checkpoint_dir(config: AppConfig, *, output_dir: Path, stem: str) -> Path:
    raw = config.model.checkpoint_dir.format(target=stem)
    path = resolve_path(raw, base=config.base_dir)
    if path is None:
        return output_dir / "model_weights" / stem
    if path.suffix.lower() == ".pt":
        raise ValueError("model.checkpoint_dir must be a directory, not a .pt file")
    if "{target}" not in config.model.checkpoint_dir:
        path = path / stem
    return path


def _component_checkpoint_path(checkpoint_dir: Path, component_name: str) -> Path:
    safe_name = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in component_name
    )
    return checkpoint_dir / f"{safe_name}.pt"


def _load_forecasters_for_components(
    decomposition: DecompositionResult,
    *,
    config: AppConfig,
    device,
    checkpoint_dir: Path,
) -> list[ComponentForecaster]:
    forecasters = []
    for component in decomposition.components:
        checkpoint_path = _component_checkpoint_path(checkpoint_dir, component.name)
        forecaster = load_component_forecaster(
            checkpoint_path,
            config=config,
            device=device,
        )
        if forecaster.name != component.name:
            raise ValueError(
                f"Checkpoint {checkpoint_path} contains component {forecaster.name!r}, "
                f"expected {component.name!r}"
            )
        forecasters.append(forecaster)
    return forecasters


def _save_forecaster_checkpoints(
    forecasters: list[ComponentForecaster],
    *,
    checkpoint_dir: Path,
    config: AppConfig,
    target: str,
) -> dict[str, str]:
    checkpoint_paths: dict[str, str] = {}
    for forecaster in forecasters:
        checkpoint_path = save_component_forecaster(
            forecaster,
            _component_checkpoint_path(checkpoint_dir, forecaster.name),
            config=config,
            target=target,
        )
        checkpoint_paths[forecaster.name] = str(checkpoint_path)
    return checkpoint_paths


def _attach_checkpoint_paths(
    component_records: list[dict],
    checkpoint_paths: dict[str, str | None],
) -> None:
    for record in component_records:
        record["checkpoint_path"] = checkpoint_paths.get(record["name"])


def _checkpoint_summary(config: AppConfig, checkpoint_dir: Path) -> dict:
    return {
        "mode": "train_and_save" if config.model.retrain_model else "load_saved_weights",
        "directory": str(checkpoint_dir),
        "target_transform": _target_transform(config),
    }


def _param_cache_path(config: AppConfig, *, output_dir: Path, stem: str) -> Path:
    if config.model.param_cache_path:
        raw = config.model.param_cache_path.format(target=stem)
        path = resolve_path(raw, base=config.base_dir)
        if path is None:
            return output_dir / f"best_params_{stem}.json"
        if path.suffix.lower() != ".json":
            return path / f"best_params_{stem}.json"
        return path
    return output_dir / f"best_params_{stem}.json"


def _load_cached_params(path: Path, *, config: AppConfig) -> dict[str, dict[str, int | float]]:
    if config.model.search_hyperparameters or not config.model.use_cached_params or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    expected = {
        "decomposition_mode": config.decomposition.mode,
        "decomposition_scope": config.decomposition.scope,
        "model_variant": config.model.variant,
        "target_transform": _target_transform(config),
        "time_scale": config.experiment.time_scale,
        "window_size": config.experiment.window_size,
        "features_enabled": bool(config.features.enabled),
        "features_use_for_res": bool(config.features.use_for_res),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            return {}
    if config.decomposition.scope == "train_only_recursive":
        expected_schema = (
            "fit_only_recursive_exog_v2" if config.features.enabled else "fit_only_recursive_v2"
        )
        if payload.get("cache_schema") != expected_schema:
            return {}
    if config.decomposition.scope == "walk_forward":
        expected_schema = "walk_forward_exog_v2" if config.features.enabled else "walk_forward_v2"
        if payload.get("cache_schema") != expected_schema:
            return {}
    components = payload.get("components", {})
    if isinstance(components, list):
        return {
            str(item["name"]): item.get("params", {})
            for item in components
            if isinstance(item, dict) and "name" in item
        }
    if isinstance(components, dict):
        result: dict[str, dict[str, int | float]] = {}
        for name, item in components.items():
            if isinstance(item, dict) and "params" in item:
                result[str(name)] = item["params"]
            elif isinstance(item, dict):
                result[str(name)] = item
        return result
    return {}


def _save_best_params(
    path: Path,
    *,
    target: str,
    decomposition: DecompositionResult,
    config: AppConfig,
    component_records: list[dict],
    selection: dict | None = None,
    split: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_schema": (
            ("walk_forward_exog_v2" if config.features.enabled else "walk_forward_v2")
            if config.decomposition.scope == "walk_forward"
            else (
                (
                    "fit_only_recursive_exog_v2"
                    if config.features.enabled
                    else "fit_only_recursive_v2"
                )
                if config.decomposition.scope == "train_only_recursive"
                else "full_sample_v1"
            )
        ),
        "target": target,
        "decomposition_mode": config.decomposition.mode,
        "decomposition_scope": config.decomposition.scope,
        "model_variant": config.model.variant,
        "target_transform": _target_transform(config),
        "time_scale": config.experiment.time_scale,
        "window_size": config.experiment.window_size,
        "features_enabled": bool(config.features.enabled),
        "features_use_for_res": bool(config.features.use_for_res),
        "vmd": {
            "k": decomposition.vmd_k,
            "alpha": decomposition.vmd_alpha,
            "pso_fitness": decomposition.pso_fitness,
        },
        "parameter_selection": selection,
        "split": split,
        "components": {
            record["name"]: {
                "params": record["params"],
                "best_val_loss": record["best_val_loss"],
                "epochs_run": record["epochs_run"],
            }
            for record in component_records
        },
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def _save_component_series(
    decomposition: DecompositionResult,
    *,
    dates: pd.Series,
    path: Path,
) -> Path:
    frame = pd.DataFrame({"Date": pd.to_datetime(dates).dt.strftime("%Y-%m-%d").to_numpy()})
    expected = len(frame)
    for component in decomposition.components:
        if len(component.values) != expected:
            raise RuntimeError(
                f"Component {component.name} has {len(component.values)} rows, expected {expected}"
            )
        frame[component.name] = component.values
    frame.to_csv(path, index=False)
    return path


def _save_fit_plot(frame: pd.DataFrame, *, target: str, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    x = pd.to_datetime(frame["Date"])
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, frame["Actual"], label="Actual", linewidth=1.5)
    ax.plot(x, frame["Predicted"], label="Predicted", linewidth=1.2)
    if target == "High" and "HighBound" in frame.columns:
        ax.fill_between(
            x,
            frame["Predicted"].astype(float),
            frame["HighBound"].astype(float),
            alpha=0.18,
            label="ACI upper band",
        )
        ax.plot(x, frame["HighBound"], label="HighBound", linewidth=1.0, linestyle="--")
    elif target == "Low" and "LowBound" in frame.columns:
        ax.fill_between(
            x,
            frame["LowBound"].astype(float),
            frame["Predicted"].astype(float),
            alpha=0.18,
            label="ACI lower band",
        )
        ax.plot(x, frame["LowBound"], label="LowBound", linewidth=1.0, linestyle="--")
    elif "LowerBound" in frame.columns and "UpperBound" in frame.columns:
        ax.fill_between(
            x,
            frame["LowerBound"].astype(float),
            frame["UpperBound"].astype(float),
            alpha=0.18,
            label="ACI interval",
        )
    ax.set_title(f"S&P 500 {target} prediction")
    ax.set_xlabel("Date")
    ax.set_ylabel(target)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_decomposition_summary(decomposition: DecompositionResult, path: Path) -> None:
    arrays = {
        "residual": decomposition.residual,
        "vmd_modes": np.asarray(decomposition.vmd_modes),
        "imfs": np.asarray(decomposition.imfs),
    }
    np.savez_compressed(path, **arrays)


def _vmd_reconstruction_error(decomposition: DecompositionResult) -> float | None:
    if not decomposition.imfs or not decomposition.vmd_modes:
        return None
    return float(
        np.linalg.norm(decomposition.imfs[0] - np.sum(decomposition.vmd_modes, axis=0))
        / (np.linalg.norm(decomposition.imfs[0]) + 1.0e-12)
    )
