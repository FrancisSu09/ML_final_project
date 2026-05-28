from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sp500_forecast.config import AppConfig, load_config
from sp500_forecast.data import load_price_data


def test_config_relative_data_path_resolves_from_project_root(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    config_dir = project / "configs"
    data_dir = project / "data" / "raw"
    unrelated = tmp_path / "unrelated"
    config_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    unrelated.mkdir()

    pd.DataFrame(
        {
            "Date": pd.date_range("2023-01-02", periods=4, freq="D"),
            "High": [101.0, 102.5, 103.0, 104.0],
            "Low": [99.5, 100.0, 101.0, 102.0],
        }
    ).to_csv(data_dir / "prices.csv", index=False)
    config_path = config_dir / "sp500.yaml"
    config_path.write_text(
        """
data:
  csv_path: data/raw/prices.csv
  start: "2023-01-02"
  end: "2023-01-06"
  sample_count:
  strict_paper_sample: false
outputs:
  directory: outputs/check
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.chdir(unrelated)

    config = load_config(config_path)
    frame = load_price_data(config)

    assert config.base_dir == project
    assert len(frame) == 4
    assert frame["High"].tolist() == [101.0, 102.5, 103.0, 104.0]


def test_investing_price_and_open_columns_are_preserved(tmp_path: Path) -> None:
    csv_path = tmp_path / "investing.csv"
    pd.DataFrame(
        {
            "Date": ["01/02/2023", "01/03/2023"],
            "Price": ["3,900.25", "3,920.50"],
            "Open": ["3,880.00", "3,901.00"],
            "High": ["3,910.00", "3,930.00"],
            "Low": ["3,870.00", "3,895.00"],
            "Vol.": ["1.5M", "2.0M"],
        }
    ).to_csv(csv_path, index=False)
    config = AppConfig()
    config.base_dir = tmp_path
    config.data.csv_path = "investing.csv"
    config.data.start = "2023-01-02"
    config.data.end = "2023-01-04"
    config.data.sample_count = None
    config.data.strict_paper_sample = False

    frame = load_price_data(config)

    assert frame["Close"].tolist() == [3900.25, 3920.5]
    assert frame["Open"].tolist() == [3880.0, 3901.0]
    assert frame["Volume"].tolist() == [1_500_000.0, 2_000_000.0]


def test_empty_csv_error_mentions_path(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("", encoding="utf-8")
    config = AppConfig()
    config.base_dir = tmp_path
    config.data.csv_path = "empty.csv"

    with pytest.raises(ValueError, match=str(csv_path)):
        load_price_data(config)
