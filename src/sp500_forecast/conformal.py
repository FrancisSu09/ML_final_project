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
    bound_band_coverage: float | None
    mean_width: float | None
    coverage_definition: str
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
    """Add volatility-normalized one-sided adaptive conformal bounds.

    Scores keep the sign of the residual, (actual - prediction), divided by a
    causal rolling volatility estimate. High uses the upper residual tail;
    Low uses the lower residual tail. ACI adapts the effective one-sided
    miscoverage level online after each observed target.
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
            bound_band_coverage=None,
            mean_width=None,
            coverage_definition="disabled",
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
    tail = _target_tail(target)
    scores = list(calibration_scores)
    lower: list[float] = []
    upper: list[float] = []
    q_values: list[float] = []
    alpha_values: list[float] = []
    signed_standardized_errors: list[float] = []
    standardized_abs_errors: list[float] = []
    bound_offsets: list[float] = []
    covered: list[bool] = []
    band_lower: list[float] = []
    band_upper: list[float] = []
    band_covered: list[bool] = []

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
            tail=tail,
        )
        q_values.append(q_t)
        offset_t = q_t * vol_t
        bound_t = pred_t + offset_t
        bound_offsets.append(offset_t)
        band_lo_t = min(pred_t, bound_t)
        band_hi_t = max(pred_t, bound_t)
        band_lower.append(band_lo_t)
        band_upper.append(band_hi_t)
        band_covered.append(bool(band_lo_t <= y_t <= band_hi_t))

        if tail == "upper":
            lo_t = np.nan
            hi_t = bound_t
            is_covered = bool(y_t <= hi_t)
        else:
            lo_t = bound_t
            hi_t = np.nan
            is_covered = bool(y_t >= lo_t)
        lower.append(lo_t)
        upper.append(hi_t)
        covered.append(is_covered)

        score_t = (y_t - pred_t) / vol_t
        signed_standardized_errors.append(score_t)
        standardized_abs_errors.append(abs(score_t))
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
    enriched["SignedStandardizedError"] = signed_standardized_errors
    enriched["StandardizedAbsError"] = standardized_abs_errors
    enriched["ACIAlpha"] = alpha_values
    enriched["ConformalQ"] = q_values
    enriched["ConformalTail"] = tail
    enriched["ConformalBoundOffset"] = bound_offsets
    enriched["BoundBandLower"] = band_lower
    enriched["BoundBandUpper"] = band_upper
    enriched["BoundBandCovered"] = band_covered
    enriched["LowerBound"] = lower
    enriched["UpperBound"] = upper
    enriched["Covered"] = covered
    if target == "High":
        enriched["HighBound"] = enriched["UpperBound"]
    elif target == "Low":
        enriched["LowBound"] = enriched["LowerBound"]

    bound_widths = np.abs(np.asarray(bound_offsets, dtype=float))
    summary = ConformalSummary(
        enabled=True,
        method=f"one-sided ACI with volatility-normalized signed residual scores ({tail} tail)",
        target_coverage=config.target_coverage,
        rolling_window=config.rolling_window,
        calibration_window=config.calibration_window,
        min_calibration_points=config.min_calibration_points,
        gamma=config.gamma,
        calibration_points=len(calibration_scores),
        realized_coverage=float(np.mean(covered)) if covered else None,
        bound_band_coverage=float(np.mean(band_covered)) if band_covered else None,
        mean_width=float(np.mean(bound_widths)) if len(bound_widths) else None,
        coverage_definition=(
            "realized_coverage is one-sided ACI coverage; "
            "bound_band_coverage is P(Actual lies between Predicted and the target bound)"
        ),
        reference=reference,
    )
    return enriched, summary


def _target_tail(target: str) -> str:
    if target == "High":
        return "upper"
    if target == "Low":
        return "lower"
    raise ValueError(f"One-sided ACI only supports High or Low targets, got {target!r}")


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
    tail: str = "upper",
) -> float:
    clean = np.asarray(scores, dtype=float)
    clean = clean[np.isfinite(clean)]
    if tail == "lower":
        return -conformal_quantile(
            -clean,
            alpha=alpha,
            fallback=-fallback,
            min_points=min_points,
            tail="upper",
        )
    if tail != "upper":
        raise ValueError(f"Unsupported conformal quantile tail: {tail}")
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
    scores = (actual - predicted) / volatility
    return [float(score) for score in scores if np.isfinite(score)]
