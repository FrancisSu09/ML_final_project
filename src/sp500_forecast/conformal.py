from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .config import ConformalConfig


@dataclass(slots=True)
class ConformalSummary:
    enabled: bool
    method: str
    target_coverage: float
    rolling_window: int
    calibration_window: int
    min_calibration_points: int
    gamma: float
    calibration_points: int
    realized_coverage: float | None
    mean_width: float | None
    reference: str

    def as_dict(self) -> dict:
        return asdict(self)


def add_aci_intervals(
    prediction_frame: pd.DataFrame,
    *,
    target: str,
    signal: np.ndarray,
    calibration_frame: pd.DataFrame | None,
    config: ConformalConfig,
) -> tuple[pd.DataFrame, ConformalSummary]:
    """Add volatility-normalized adaptive conformal intervals to predictions.

    Scores are |actual - prediction| divided by a causal rolling volatility
    estimate. ACI then adapts the effective miscoverage level online after each
    observed target.
    """
    reference = "Gibbs and Candes (2021), Adaptive Conformal Inference Under Distribution Shift"
    if not config.enabled:
        return prediction_frame, ConformalSummary(
            enabled=False,
            method="disabled",
            target_coverage=config.target_coverage,
            rolling_window=config.rolling_window,
            calibration_window=config.calibration_window,
            min_calibration_points=config.min_calibration_points,
            gamma=config.gamma,
            calibration_points=0,
            realized_coverage=None,
            mean_width=None,
            reference=reference,
        )

    target_indices = prediction_frame["TargetIndex"].to_numpy(dtype=int)
    volatility = causal_rolling_volatility(
        signal,
        target_indices=target_indices,
        rolling_window=config.rolling_window,
        floor=config.volatility_floor,
    )

    calibration_scores = _initial_scores(
        calibration_frame=calibration_frame,
        signal=signal,
        config=config,
    )
    alpha = float(np.clip(1.0 - config.target_coverage, config.min_alpha, config.max_alpha))
    scores = list(calibration_scores)
    lower: list[float] = []
    upper: list[float] = []
    q_values: list[float] = []
    alpha_values: list[float] = []
    standardized_errors: list[float] = []
    covered: list[bool] = []

    actual = prediction_frame["Actual"].to_numpy(dtype=float)
    predicted = prediction_frame["Predicted"].to_numpy(dtype=float)
    for y_t, pred_t, vol_t in zip(actual, predicted, volatility, strict=True):
        alpha_values.append(alpha)
        recent_scores = scores[-config.calibration_window :] if config.calibration_window > 0 else scores
        q_t = conformal_quantile(
            recent_scores,
            alpha=alpha,
            fallback=0.0,
            min_points=config.min_calibration_points,
        )
        q_values.append(q_t)
        width_t = q_t * vol_t
        lo_t = pred_t - width_t
        hi_t = pred_t + width_t
        lower.append(lo_t)
        upper.append(hi_t)
        is_covered = bool(lo_t <= y_t <= hi_t)
        covered.append(is_covered)

        score_t = abs(y_t - pred_t) / vol_t
        standardized_errors.append(score_t)
        scores.append(float(score_t))
        alpha = float(
            np.clip(
                alpha + config.gamma * ((1.0 - config.target_coverage) - float(not is_covered)),
                config.min_alpha,
                config.max_alpha,
            )
        )

    enriched = prediction_frame.copy()
    enriched["Volatility"] = volatility
    enriched["StandardizedAbsError"] = standardized_errors
    enriched["ACIAlpha"] = alpha_values
    enriched["ConformalQ"] = q_values
    enriched["LowerBound"] = lower
    enriched["UpperBound"] = upper
    enriched["Covered"] = covered
    if target == "High":
        enriched["HighBound"] = enriched["UpperBound"]
    elif target == "Low":
        enriched["LowBound"] = enriched["LowerBound"]

    widths = enriched["UpperBound"] - enriched["LowerBound"]
    summary = ConformalSummary(
        enabled=True,
        method="ACI with volatility-normalized absolute residual scores",
        target_coverage=config.target_coverage,
        rolling_window=config.rolling_window,
        calibration_window=config.calibration_window,
        min_calibration_points=config.min_calibration_points,
        gamma=config.gamma,
        calibration_points=len(calibration_scores),
        realized_coverage=float(np.mean(covered)) if covered else None,
        mean_width=float(np.mean(widths)) if len(widths) else None,
        reference=reference,
    )
    return enriched, summary


def causal_rolling_volatility(
    signal: np.ndarray,
    *,
    target_indices: np.ndarray,
    rolling_window: int,
    floor: float,
) -> np.ndarray:
    values = np.asarray(signal, dtype=float)
    vol = []
    window = max(2, int(rolling_window))
    for target_index in np.asarray(target_indices, dtype=int):
        history = values[:target_index]
        changes = np.diff(history[-(window + 1) :])
        if len(changes) >= 2:
            scale = float(np.std(changes, ddof=1))
        elif len(changes) == 1:
            scale = float(abs(changes[0]))
        else:
            scale = floor
        vol.append(max(scale, floor))
    return np.asarray(vol, dtype=float)


def conformal_quantile(
    scores: list[float] | np.ndarray,
    *,
    alpha: float,
    fallback: float,
    min_points: int,
) -> float:
    clean = np.asarray(scores, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return float(fallback)
    if len(clean) < min_points:
        return float(np.max(clean))
    level = min(1.0, np.ceil((len(clean) + 1) * (1.0 - alpha)) / len(clean))
    return float(np.quantile(clean, level, method="higher"))


def _initial_scores(
    *,
    calibration_frame: pd.DataFrame | None,
    signal: np.ndarray,
    config: ConformalConfig,
) -> list[float]:
    if calibration_frame is None or calibration_frame.empty:
        return []
    indices = calibration_frame["TargetIndex"].to_numpy(dtype=int)
    volatility = causal_rolling_volatility(
        signal,
        target_indices=indices,
        rolling_window=config.rolling_window,
        floor=config.volatility_floor,
    )
    actual = calibration_frame["Actual"].to_numpy(dtype=float)
    predicted = calibration_frame["Predicted"].to_numpy(dtype=float)
    scores = np.abs(actual - predicted) / volatility
    return [float(score) for score in scores if np.isfinite(score)]
