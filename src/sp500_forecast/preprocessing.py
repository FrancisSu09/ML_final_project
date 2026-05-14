from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPS = 1.0e-12


@dataclass(slots=True)
class MinMaxScaler1D:
    data_min_: float | None = None
    data_max_: float | None = None

    def fit(self, values: np.ndarray) -> "MinMaxScaler1D":
        values = np.asarray(values, dtype=float)
        self.data_min_ = float(np.min(values))
        self.data_max_ = float(np.max(values))
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.data_min_ is None or self.data_max_ is None:
            raise RuntimeError("Scaler has not been fitted")
        values = np.asarray(values, dtype=float)
        span = max(self.data_max_ - self.data_min_, EPS)
        return (values - self.data_min_) / span

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        if self.data_min_ is None or self.data_max_ is None:
            raise RuntimeError("Scaler has not been fitted")
        values = np.asarray(values, dtype=float)
        return values * (self.data_max_ - self.data_min_) + self.data_min_


def make_supervised(series: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create rolling-window X, y, and target indices for one-step prediction."""
    series = np.asarray(series, dtype=float)
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if len(series) <= window_size:
        raise ValueError("series length must be greater than window_size")

    xs: list[np.ndarray] = []
    ys: list[float] = []
    indices: list[int] = []
    for end in range(window_size, len(series)):
        xs.append(series[end - window_size : end])
        ys.append(float(series[end]))
        indices.append(end)
    x_arr = np.asarray(xs, dtype=float)[..., None]
    y_arr = np.asarray(ys, dtype=float)[:, None]
    return x_arr, y_arr, np.asarray(indices, dtype=int)


def train_val_test_masks(
    target_indices: np.ndarray,
    *,
    series_length: int,
    train_ratio: float,
    validation_ratio: float,
    split_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if split_index is None:
        split_index = int(series_length * train_ratio)
    split_index = int(np.clip(split_index, 1, series_length - 1))
    train_mask = target_indices < split_index
    test_mask = target_indices >= split_index

    train_positions = np.flatnonzero(train_mask)
    if len(train_positions) == 0:
        raise ValueError("No training windows. Increase data length or lower window_size.")
    val_count = max(1, int(len(train_positions) * validation_ratio))
    val_positions = train_positions[-val_count:]
    fit_positions = train_positions[:-val_count]
    if len(fit_positions) == 0:
        fit_positions = train_positions
        val_positions = train_positions[-1:]

    fit_mask = np.zeros_like(train_mask, dtype=bool)
    val_mask = np.zeros_like(train_mask, dtype=bool)
    fit_mask[fit_positions] = True
    val_mask[val_positions] = True
    return fit_mask, val_mask, test_mask
