from __future__ import annotations

import numpy as np
import pandas as pd

from sp500_forecast.config import ConformalConfig
from sp500_forecast.conformal import (
    add_aci_intervals,
    causal_rolling_volatility,
    conformal_quantile,
)


def test_causal_rolling_volatility_uses_only_previous_values() -> None:
    signal = np.array([10.0, 11.0, 13.0, 16.0, 20.0])
    vol = causal_rolling_volatility(
        signal,
        target_indices=np.array([3, 4]),
        rolling_window=2,
        floor=1.0e-8,
    )

    np.testing.assert_allclose(vol, [np.std([1.0, 2.0], ddof=1), np.std([2.0, 3.0], ddof=1)])


def test_high_aci_uses_signed_upper_tail_scores() -> None:
    signal = np.array([100.0, 101.0, 103.0, 106.0, 110.0, 115.0, 121.0, 128.0])
    calibration = pd.DataFrame(
        {
            "TargetIndex": [3, 4, 5],
            "Actual": [106.0, 110.0, 115.0],
            "Predicted": [107.0, 109.0, 114.0],
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
    assert "SignedStandardizedError" in enriched.columns
    assert "HighBound" in enriched.columns
    assert np.isnan(enriched.loc[0, "LowerBound"])
    assert enriched.loc[0, "ConformalQ"] > 0
    assert enriched.loc[0, "UpperBound"] == enriched.loc[0, "HighBound"]
    assert enriched.loc[0, "Covered"]
    assert enriched.loc[0, "BoundBandCovered"]
    assert summary.bound_band_coverage == 1.0
    assert summary.coverage_definition.startswith("realized_coverage is one-sided")
    assert enriched.loc[0, "SignedStandardizedError"] == (121.0 - 120.0) / enriched.loc[0, "Volatility"]


def test_low_aci_uses_signed_lower_tail_scores() -> None:
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
        target="Low",
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
    assert "LowBound" in enriched.columns
    assert np.isnan(enriched.loc[0, "UpperBound"])
    assert enriched.loc[0, "ConformalQ"] > 0
    assert enriched.loc[0, "LowerBound"] == enriched.loc[0, "LowBound"]
    assert enriched.loc[0, "LowBound"] > enriched.loc[0, "Predicted"]
    assert enriched.loc[0, "Covered"]
    assert enriched.loc[0, "BoundBandCovered"]
    assert summary.bound_band_coverage == 1.0


def test_bound_band_coverage_can_differ_from_one_sided_coverage() -> None:
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
            "Actual": [119.0],
            "Predicted": [120.0],
            "NaivePreviousValue": [115.0],
            "Error": [1.0],
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

    assert enriched.loc[0, "Covered"]
    assert not enriched.loc[0, "BoundBandCovered"]
    assert summary.realized_coverage == 1.0
    assert summary.bound_band_coverage == 0.0


def test_conformal_quantile_lower_tail_preserves_direction() -> None:
    scores = np.array([-2.0, -1.0, 0.5, 3.0])

    assert conformal_quantile(scores, alpha=0.2, fallback=0.0, min_points=1, tail="lower") < 0
