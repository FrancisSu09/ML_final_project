from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FeatureConfig


EPS = 1.0e-12


def build_feature_frame(frame: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Build causal, mostly stationary exogenous features aligned to price rows."""
    if not config.enabled:
        return pd.DataFrame(index=frame.index)

    features: dict[str, pd.Series] = {}
    close = _series_or_midpoint(frame)
    high = pd.to_numeric(frame["High"], errors="coerce")
    low = pd.to_numeric(frame["Low"], errors="coerce")
    open_ = _optional_numeric(frame, "Open")
    volume = _optional_numeric(frame, "Volume")
    prev_close = close.shift(1)

    if config.include_ohlc:
        features["CloseReturn"] = _pct_change(close, prev_close)
        features["LogCloseReturn"] = np.log(close.clip(lower=EPS)).diff()
        features["HighPrevClosePct"] = _pct_change(high, prev_close)
        features["LowPrevClosePct"] = _pct_change(low, prev_close)
        features["HighLowRangePct"] = _safe_div(high - low, prev_close)
        if open_ is not None:
            features["OpenPrevClosePct"] = _pct_change(open_, prev_close)
            features["CloseOpenChangePct"] = _safe_div(close - open_, open_)
    if config.include_volume and volume is not None and volume.notna().any():
        features["VolumeChangePct"] = volume.pct_change()
        features["LogVolumeChange"] = np.log(volume.clip(lower=EPS)).diff()

    if config.include_technical:
        returns = close.pct_change()
        for span in (5, 10, 20):
            ma = close.rolling(span, min_periods=1).mean()
            features[f"CloseSMA{span}Ratio"] = _pct_change(close, ma)
            features[f"ReturnStd{span}"] = returns.rolling(span, min_periods=2).std()

        features["RSI14"] = _rsi(close, period=14)
        macd, signal, hist = _macd(close)
        features["MACDPct"] = _safe_div(macd, close)
        features["MACDSignalPct"] = _safe_div(signal, close)
        features["MACDHistPct"] = _safe_div(hist, close)
        features["RollingVolatility20"] = returns.rolling(20, min_periods=2).std()
        features["BollingerBandwidth20"] = _bollinger_bandwidth(close, period=20)

    feature_frame = pd.DataFrame(features, index=frame.index)
    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan)
    feature_frame = feature_frame.fillna(0.0)
    return feature_frame.astype(float)


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _pct_change(values: pd.Series, baseline: pd.Series) -> pd.Series:
    return _safe_div(values, baseline) - 1.0


def _series_or_midpoint(frame: pd.DataFrame) -> pd.Series:
    close = _optional_numeric(frame, "Close")
    if close is not None:
        return close
    return (
        pd.to_numeric(frame["High"], errors="coerce")
        + pd.to_numeric(frame["Low"], errors="coerce")
    ) / 2.0


def _optional_numeric(frame: pd.DataFrame, column: str) -> pd.Series | None:
    if column not in frame.columns:
        return None
    return pd.to_numeric(frame[column], errors="coerce")


def _rsi(close: pd.Series, *, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=1).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = close.ewm(span=12, adjust=False, min_periods=1).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=1).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False, min_periods=1).mean()
    return macd, signal, macd - signal



def _bollinger_bandwidth(close: pd.Series, *, period: int) -> pd.Series:
    ma = close.rolling(period, min_periods=1).mean()
    std = close.rolling(period, min_periods=2).std()
    upper = ma + 2.0 * std
    lower = ma - 2.0 * std
    return (upper - lower) / ma.replace(0, np.nan)
