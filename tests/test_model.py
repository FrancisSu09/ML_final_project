from __future__ import annotations

import warnings

import numpy as np
import torch

from sp500_forecast.config import AppConfig
from sp500_forecast.model import BiLSTMSAMTCN
from sp500_forecast.preprocessing import MinMaxScaler1D
from sp500_forecast.training import (
    ComponentForecaster,
    load_component_forecaster,
    save_component_forecaster,
)


def test_model_forward_shape_without_weight_norm_deprecation() -> None:
    model = BiLSTMSAMTCN(
        input_size=1,
        hidden_size=4,
        tcn_channels=4,
        tcn_kernel_size=2,
        tcn_dilations=[1, 2],
        dropout=0.0,
    )
    x = torch.randn(3, 5, 1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        y = model(x)

    assert y.shape == (3, 1)
    assert not any("weight_norm is deprecated" in str(item.message) for item in caught)


def test_component_forecaster_checkpoint_round_trip(tmp_path) -> None:
    config = AppConfig()
    config.model.hidden_size = 4
    config.model.tcn_channels = 4
    config.model.tcn_kernel_size = 2
    config.model.tcn_dilations = [1, 2]
    config.model.dropout = 0.0
    params = {
        "hidden_size": 4,
        "epochs": 2,
        "batch_size": 2,
        "learning_rate": 0.001,
    }
    model = BiLSTMSAMTCN(
        input_size=1,
        hidden_size=4,
        tcn_channels=4,
        tcn_kernel_size=2,
        tcn_dilations=[1, 2],
        dropout=0.0,
    )
    model.eval()
    forecaster = ComponentForecaster(
        name="IMF2",
        model=model,
        scaler=MinMaxScaler1D(data_min_=0.0, data_max_=10.0),
        feature_scaler=None,
        best_val_loss=0.25,
        params=params,
        epochs_run=2,
    )

    checkpoint = save_component_forecaster(
        forecaster,
        tmp_path / "IMF2.pt",
        config=config,
        target="High",
    )
    loaded = load_component_forecaster(checkpoint, config=config, device=torch.device("cpu"))
    x = torch.randn(2, 3, 1)

    with torch.no_grad():
        expected = model(x).numpy()
        actual = loaded.model(x).numpy()

    assert loaded.name == "IMF2"
    assert loaded.params == params
    assert loaded.scaler.data_min_ == 0.0
    assert loaded.scaler.data_max_ == 10.0
    np.testing.assert_allclose(actual, expected)
