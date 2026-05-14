from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import AppConfig, resolve_path
from .data import load_price_data
from .decomposition import DecompositionResult, decompose_signal
from .metrics import modified_diebold_mariano, regression_metrics
from .training import ComponentPrediction, choose_device, train_component


@dataclass(slots=True)
class TargetRunResult:
    target: str
    prediction_path: Path
    summary_path: Path
    component_path: Path | None
    plot_path: Path | None
    metrics: dict[str, float]


def run_pipeline(config: AppConfig, *, targets: list[str] | None = None) -> list[TargetRunResult]:
    frame = load_price_data(config)
    frame = apply_time_scale(frame, config.experiment.time_scale)
    selected_targets = normalize_targets(targets or config.experiment.targets)
    output_dir = resolve_path(config.outputs.directory)
    if output_dir is None:
        raise ValueError("outputs.directory cannot be empty")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for target in selected_targets:
        results.append(_run_one_target(frame, target=target, config=config, output_dir=output_dir))
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
    scaled = (
        frame.assign(_group=grouped)
        .groupby("_group", as_index=False)
        .agg({"Date": "last", "High": "max", "Low": "min"})
    )
    return scaled[["Date", "High", "Low"]].reset_index(drop=True)


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
) -> TargetRunResult:
    signal = frame[target].to_numpy(dtype=float)
    dates = pd.to_datetime(frame["Date"]).reset_index(drop=True)
    device = choose_device(config.experiment.device)
    split_index = split_index_from_date(dates, config.experiment.train_split_date)

    decomposition = decompose_signal(signal, config.decomposition, target=target)
    component_predictions: list[ComponentPrediction] = []
    for component in decomposition.components:
        component_predictions.append(
            train_component(
                name=component.name,
                series=component.values,
                config=config,
                device=device,
                split_index=split_index,
            )
        )

    target_indices = component_predictions[0].target_indices
    for prediction in component_predictions[1:]:
        if not np.array_equal(target_indices, prediction.target_indices):
            raise RuntimeError("Component predictions are not aligned on the same target indices")

    predicted = np.sum([prediction.predicted for prediction in component_predictions], axis=0)
    actual = signal[target_indices]
    naive = signal[target_indices - 1]
    metrics = regression_metrics(actual, predicted)
    naive_metrics = regression_metrics(actual, naive)
    mdm = {
        loss: asdict(modified_diebold_mariano(actual, predicted, naive, loss=loss))
        for loss in ("mape", "mae", "rmse")
    }

    stem = target.lower()
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

    summary = {
        "target": target,
        "rows": int(len(signal)),
        "test_rows": int(len(actual)),
        "paper_method": "ICEEMDAN-PSO-VMD-BiLSTM-SAM-TCN",
        "decomposition_mode": config.decomposition.mode,
        "model_variant": config.model.variant,
        "time_scale": config.experiment.time_scale,
        "train_split_date": config.experiment.train_split_date,
        "train_split_index": split_index,
        "metrics": metrics.as_dict(),
        "naive_previous_value_metrics": naive_metrics.as_dict(),
        "mdm_vs_naive": mdm,
        "vmd": {
            "k": decomposition.vmd_k,
            "alpha": decomposition.vmd_alpha,
            "pso_fitness": decomposition.pso_fitness,
            "imf1_reconstruction_error": _vmd_reconstruction_error(decomposition),
        },
        "components": [
            {
                "name": prediction.name,
                "best_val_loss": prediction.best_val_loss,
                "params": prediction.params,
                "epochs_run": prediction.epochs_run,
            }
            for prediction in component_predictions
        ],
    }
    summary_path = output_dir / f"summary_{stem}.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    _save_decomposition_summary(decomposition, output_dir / f"decomposition_{stem}.npz")

    return TargetRunResult(
        target=target,
        prediction_path=prediction_path,
        summary_path=summary_path,
        component_path=component_path,
        plot_path=plot_path,
        metrics=metrics.as_dict(),
    )


def _save_fit_plot(frame: pd.DataFrame, *, target: str, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    x = pd.to_datetime(frame["Date"])
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, frame["Actual"], label="Actual", linewidth=1.5)
    ax.plot(x, frame["Predicted"], label="Predicted", linewidth=1.2)
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
