from __future__ import annotations

import numpy as np
import pandas as pd

from sp500_forecast.config import ConformalConfig
from sp500_forecast.conformal import add_aci_intervals, causal_rolling_volatility


def test_causal_rolling_volatility_uses_only_previous_values() -> None:
    signal = np.array([10.0, 11.0, 13.0, 16.0, 20.0])
    vol = causal_rolling_volatility(
        signal,
        target_indices=np.array([3, 4]),
        rolling_window=2,
        floor=1.0e-8,
    )

    np.testing.assert_allclose(vol, [np.std([1.0, 2.0], ddof=1), np.std([2.0, 3.0], ddof=1)])


def test_aci_intervals_use_standardized_calibration_scores() -> None:
    signal = np.array([100.0, 101.0, 103.0, 106.0, 110.0, 115.0, 121.0, 128.0])
    calibration = pd.DataFrame(
        {
            "TargetIndex": [3, 4, 5],
            "Actual": [106.0, 110.0, 115.0],
            "Predicted": [105.0, 109.0, 114.0],
        }
    )
    predictions = pd.DataFrame(
        {
            "Date": ["2023-01-09"],
            "TargetIndex": [6],
            "Actual": [121.0],
            "Predicted": [120.0],
            "NaivePreviousValue": [115.0],
            "Error": [-1.0],
        }
    )

    enriched, summary = add_aci_intervals(
        predictions,
        target="High",
        signal=signal,
        calibration_frame=calibration,
        config=ConformalConfig(
            target_coverage=0.8,
            rolling_window=3,
            min_calibration_points=1,
            gamma=0.01,
        ),
    )

    assert summary.calibration_points == 3
    assert "HighBound" in enriched.columns
    assert enriched.loc[0, "ConformalQ"] > 0
    assert enriched.loc[0, "UpperBound"] == enriched.loc[0, "HighBound"]
    assert enriched.loc[0, "StandardizedAbsError"] == abs(121.0 - 120.0) / enriched.loc[0, "Volatility"]
