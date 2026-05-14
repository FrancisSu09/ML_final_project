from __future__ import annotations

import numpy as np

from sp500_forecast.decomposition import Component, DecompositionResult
from sp500_forecast.pipeline import _align_walk_forward_components


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
