from __future__ import annotations

import numpy as np

from sp500_forecast.preprocessing import make_supervised


def test_feature_windows_exclude_target_row() -> None:
    series = np.arange(6, dtype=float)
    features = np.arange(60, dtype=float).reshape(6, 10)

    x, y, target_indices = make_supervised(series, window_size=3, features=features)

    assert target_indices[0] == 3
    assert y[0, 0] == 3.0
    np.testing.assert_allclose(x[0, :, 0], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(x[0, :, 1:], features[0:3])
    assert not np.any(x[0, :, 1:] == features[3])
