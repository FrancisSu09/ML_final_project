from __future__ import annotations

import warnings

import torch

from sp500_forecast.model import BiLSTMSAMTCN


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
