from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


EPS = 1.0e-12


@dataclass(slots=True)
class MetricResult:
    mape: float
    mae: float
    rmse: float

    def as_dict(self) -> dict[str, float]:
        return {"MAPE": self.mape, "MAE": self.mae, "RMSE": self.rmse}


@dataclass(slots=True)
class MDMResult:
    statistic: float
    p_value: float | None
    loss: str
    horizon: int


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> MetricResult:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    errors = predicted - actual
    denom = np.maximum(np.abs(actual), EPS)
    return MetricResult(
        mape=float(np.mean(np.abs(errors / denom)) * 100.0),
        mae=float(np.mean(np.abs(errors))),
        rmse=float(np.sqrt(np.mean(errors**2))),
    )


def improvement_rate(other: MetricResult, proposed: MetricResult) -> dict[str, float]:
    """Return improvement percentages of proposed model versus another model."""
    values = {}
    for name in ("mape", "mae", "rmse"):
        base = getattr(other, name)
        new = getattr(proposed, name)
        values[name.upper()] = float((base - new) / max(abs(base), EPS) * 100.0)
    return values


def modified_diebold_mariano(
    actual: np.ndarray,
    model_1: np.ndarray,
    model_2: np.ndarray,
    *,
    loss: str = "mae",
    horizon: int = 1,
) -> MDMResult:
    """Harvey-Leybourne-Newbold modified Diebold-Mariano test.

    A negative statistic means model_1 has smaller average loss than model_2.
    """
    actual = np.asarray(actual, dtype=float)
    model_1 = np.asarray(model_1, dtype=float)
    model_2 = np.asarray(model_2, dtype=float)
    if not (len(actual) == len(model_1) == len(model_2)):
        raise ValueError("actual, model_1, and model_2 must have the same length")
    if len(actual) < 3:
        raise ValueError("MDM test needs at least 3 observations")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    d = _loss_series(actual, model_1, loss) - _loss_series(actual, model_2, loss)
    t = len(d)
    d_bar = float(np.mean(d))
    lrv = _long_run_variance(d, horizon=horizon)
    if lrv <= EPS:
        statistic = 0.0 if abs(d_bar) <= EPS else math.copysign(float("inf"), d_bar)
    else:
        dm = d_bar / math.sqrt(lrv / t)
        correction = math.sqrt((t + 1 - 2 * horizon + horizon * (horizon - 1) / t) / t)
        statistic = correction * dm

    p_value = None
    try:
        from scipy import stats

        p_value = float(2.0 * stats.t.sf(abs(statistic), df=t - 1))
    except Exception:
        p_value = None

    return MDMResult(
        statistic=float(statistic),
        p_value=p_value,
        loss=loss,
        horizon=horizon,
    )


def _loss_series(actual: np.ndarray, predicted: np.ndarray, loss: str) -> np.ndarray:
    err = predicted - actual
    loss = loss.lower()
    if loss in {"mae", "absolute"}:
        return np.abs(err)
    if loss in {"mse", "rmse"}:
        return err**2
    if loss == "mape":
        return np.abs(err / np.maximum(np.abs(actual), EPS)) * 100.0
    raise ValueError(f"Unsupported loss for MDM test: {loss}")


def _long_run_variance(d: np.ndarray, *, horizon: int) -> float:
    d = np.asarray(d, dtype=float)
    centered = d - np.mean(d)
    t = len(centered)
    gamma0 = float(np.dot(centered, centered) / t)
    if horizon == 1:
        return gamma0
    total = gamma0
    max_lag = min(horizon - 1, t - 1)
    for lag in range(1, max_lag + 1):
        cov = float(np.dot(centered[lag:], centered[:-lag]) / t)
        total += 2.0 * cov
    return max(total, 0.0)

