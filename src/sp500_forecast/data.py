from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import AppConfig, resolve_path


REQUIRED_PRICE_COLUMNS = ("Date", "High", "Low")


def download_sp500(config: AppConfig) -> Path:
    """Download S&P 500 OHLCV data from Yahoo Finance."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is required for downloading data. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    out_path = resolve_path(config.data.raw_output, base=config.base_dir)
    if out_path is None:
        raise ValueError("data.raw_output cannot be empty when downloading data")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frame = yf.download(
        config.data.ticker,
        start=config.data.start,
        end=config.data.end,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if frame.empty:
        raise RuntimeError(
            f"No rows returned for {config.data.ticker} "
            f"from {config.data.start} to {config.data.end}."
        )

    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [col[0] for col in frame.columns]
    frame = frame.reset_index()
    frame.to_csv(out_path, index=False)
    return out_path


def load_price_data(config: AppConfig) -> pd.DataFrame:
    """Load S&P 500 price data from CSV or download it when no CSV is provided."""
    csv_path = resolve_path(config.data.csv_path, base=config.base_dir)
    if csv_path is None:
        csv_path = resolve_path(config.data.raw_output, base=config.base_dir)
        if csv_path is None or not csv_path.exists():
            csv_path = download_sp500(config)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file does not exist: {csv_path}\n"
            "For paper replication, export S&P 500 Historical Data from "
            "Investing.com for 2011-05-02 through 2023-03-31 and save it to "
            "the configured data.csv_path. The Yahoo Finance `download` "
            "command is only a convenience fallback, not the paper source."
        )

    try:
        frame = pd.read_csv(csv_path, encoding="utf-8-sig")
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"CSV file is empty or has no columns: {csv_path}") from exc
    if frame.empty:
        raise ValueError(f"CSV file contains no rows: {csv_path}")
    frame = standardize_price_frame(frame, date_column=config.data.date_column)
    frame = frame.sort_values("Date").reset_index(drop=True)
    frame = filter_date_range(frame, start=config.data.start, end=config.data.end)

    if config.data.strict_paper_sample and config.data.sample_count:
        validate_paper_sample(frame, sample_count=config.data.sample_count)
    elif config.data.sample_count and config.data.allow_truncate and len(frame) > config.data.sample_count:
        frame = frame.tail(config.data.sample_count).reset_index(drop=True)

    return frame


def filter_date_range(frame: pd.DataFrame, *, start: str | None, end: str | None) -> pd.DataFrame:
    result = frame
    if start:
        result = result[result["Date"] >= pd.Timestamp(start)]
    if end:
        # yfinance uses an exclusive end date; applying the same convention to
        # CSV inputs keeps the paper window 2011-05-02 through 2023-03-31 when
        # end is configured as 2023-04-01.
        result = result[result["Date"] < pd.Timestamp(end)]
    return result.reset_index(drop=True)


def validate_paper_sample(frame: pd.DataFrame, *, sample_count: int) -> None:
    if len(frame) != sample_count:
        first = frame["Date"].iloc[0].date() if len(frame) else "NA"
        last = frame["Date"].iloc[-1].date() if len(frame) else "NA"
        raise ValueError(
            "Strict paper replication requires exactly "
            f"{sample_count} rows after date filtering; got {len(frame)} "
            f"from {first} to {last}. Use the Investing data exported for the "
            "paper window, or set data.strict_paper_sample=false for exploratory runs."
        )


def standardize_price_frame(frame: pd.DataFrame, *, date_column: str = "Date") -> pd.DataFrame:
    """Normalize common Yahoo/Investing CSV column variants to Date, High, Low and OHLCV."""
    if frame.empty:
        raise ValueError("Input price frame is empty")

    columns = {str(col).strip(): col for col in frame.columns}
    lower_map = {str(col).strip().lower().replace(" ", ""): col for col in frame.columns}

    date_key = date_column.lower().replace(" ", "")
    if date_key in lower_map:
        date_col = lower_map[date_key]
    elif "date" in lower_map:
        date_col = lower_map["date"]
    elif "日期" in lower_map:
        date_col = lower_map["日期"]
    elif "時間" in lower_map:
        date_col = lower_map["時間"]
    elif "时间" in lower_map:
        date_col = lower_map["时间"]
    else:
        raise ValueError(f"Cannot find date column. Available columns: {list(frame.columns)}")

    def find_price_col(name: str, aliases: tuple[str, ...]) -> str:
        key = name.lower()
        variants = [key, f"{key}price", f"{key}.", f"{key}*", *aliases]
        for variant in variants:
            if variant in lower_map:
                return lower_map[variant]
        if name in columns:
            return columns[name]
        raise ValueError(f"Cannot find {name!r} column. Available columns: {list(frame.columns)}")

    def find_optional_col(name: str, aliases: tuple[str, ...]) -> str | None:
        key = name.lower()
        variants = [key, f"{key}price", f"{key}.", f"{key}*", *aliases]
        for variant in variants:
            if variant in lower_map:
                return lower_map[variant]
        if name in columns:
            return columns[name]
        return None

    high_col = find_price_col("High", ("高", "最高", "最高價", "最高价"))
    low_col = find_price_col("Low", ("低", "最低", "最低價", "最低价"))
    open_col = find_optional_col("Open", ("openprice", "開", "开", "開盤", "开盘", "開盤價", "开盘价"))
    close_col = find_optional_col(
        "Close",
        ("price", "last", "lastprice", "closeprice", "收", "收盤", "收盘", "收盤價", "收盘价"),
    )
    volume_col = find_optional_col("Volume", ("vol", "vol.", "volume", "成交量"))

    data = {
        "Date": _to_datetime(frame[date_col]),
        "High": _to_number(frame[high_col]),
        "Low": _to_number(frame[low_col]),
    }
    if open_col is not None:
        data["Open"] = _to_number(frame[open_col])
    if close_col is not None:
        data["Close"] = _to_number(frame[close_col])
    if volume_col is not None:
        data["Volume"] = _to_number(frame[volume_col])

    result = pd.DataFrame(data)
    result = result.dropna(subset=list(REQUIRED_PRICE_COLUMNS)).reset_index(drop=True)
    if result.empty:
        raise ValueError("No usable Date/High/Low rows after cleaning the CSV")
    return result


def _to_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace("年", "-", regex=False)
        .str.replace("月", "-", regex=False)
        .str.replace("日", "", regex=False)
    )
    try:
        parsed = pd.to_datetime(cleaned, errors="coerce", format="mixed")
    except TypeError:  # pragma: no cover - pandas < 2 fallback
        parsed = pd.to_datetime(cleaned, errors="coerce")
    if parsed.isna().mean() > 0.5:
        parsed_dayfirst = pd.to_datetime(cleaned, errors="coerce", dayfirst=True)
        if parsed_dayfirst.notna().sum() > parsed.notna().sum():
            parsed = parsed_dayfirst
    return parsed


def _to_number(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    multipliers = cleaned.str[-1].str.upper().map({"K": 1e3, "M": 1e6, "B": 1e9}).fillna(1.0)
    numeric_text = cleaned.str.replace(r"([KMB])$", "", regex=True, case=False)
    return pd.to_numeric(numeric_text, errors="coerce") * multipliers
