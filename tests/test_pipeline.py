from __future__ import annotations

import numpy as np

from sp500_forecast.config import AppConfig
from sp500_forecast.decomposition import Component, DecompositionResult
from sp500_forecast.pipeline import (
    _align_walk_forward_components,
    _features_for_component,
    _restore_price_predictions,
    _target_model_signal,
)


def test_walk_forward_alignment_folds_extra_components_into_residual() -> None:
    decomposition = DecompositionResult(
        imfs=[],
        residual=np.array([10.0, 20.0, 30.0]),
        vmd_modes=[],
        vmd_k=0,
        vmd_alpha=0.0,
        pso_fitness=0.0,
        components=[
            Component("IMF2", np.array([1.0, 2.0, 3.0])),
            Component("Res", np.array([10.0, 20.0, 30.0])),
            Component("IMF8", np.array([100.0, 200.0, 300.0])),
            Component("IMF9", np.array([1000.0, 2000.0, 3000.0])),
        ],
    )

    aligned, extras, folded_target = _align_walk_forward_components(
        decomposition,
        component_names=["IMF2", "Res"],
    )

    assert extras == ["IMF8", "IMF9"]
    assert folded_target == "Res"
    np.testing.assert_allclose(aligned["IMF2"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(aligned["Res"], [1110.0, 2220.0, 3330.0])


def test_residual_component_skips_exogenous_features_by_default() -> None:
    config = AppConfig()
    config.features.enabled = True
    config.features.use_for_res = False
    features = np.ones((4, 2))

    assert _features_for_component("Res", features, config=config) is None
    np.testing.assert_allclose(
        _features_for_component("IMF2", features, config=config),
        features,
    )


def test_delta_target_transform_reconstructs_price_from_previous_value() -> None:
    config = AppConfig()
    config.experiment.target_transform = "delta"
    price = np.array([100.0, 103.0, 101.0, 108.0])

    model_signal = _target_model_signal(price, config=config)
    reconstructed = _restore_price_predictions(
        np.array([2.5, -1.0]),
        price_signal=price,
        target_indices=np.array([2, 3]),
        config=config,
    )

    np.testing.assert_allclose(model_signal, [0.0, 3.0, -2.0, 7.0])
    np.testing.assert_allclose(reconstructed, [105.5, 100.0])
