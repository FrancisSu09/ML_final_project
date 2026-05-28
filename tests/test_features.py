from __future__ import annotations

import pandas as pd

from sp500_forecast.config import FeatureConfig
from sp500_forecast.features import build_feature_frame


def test_features_use_relative_ohlc_not_raw_price_levels() -> None:
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2023-01-02", periods=4),
            "Open": [100.0, 105.0, 110.0, 120.0],
            "High": [106.0, 111.0, 121.0, 126.0],
            "Low": [99.0, 103.0, 108.0, 118.0],
            "Close": [104.0, 109.0, 119.0, 125.0],
        }
    )

    features = build_feature_frame(frame, FeatureConfig(enabled=True))

    assert "Open" not in features.columns
    assert "High" not in features.columns
    assert "Low" not in features.columns
    assert "Close" not in features.columns
    assert "OpenPrevClosePct" in features.columns
    assert "HighPrevClosePct" in features.columns
    assert "MACDPct" in features.columns
    assert "RollingVolatility20" in features.columns
