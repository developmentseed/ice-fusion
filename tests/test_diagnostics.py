import numpy as np

from fusion.diagnostics import weight_stability_across_seeds


def test_stability_across_identical_inputs_is_zero():
    weights_list = [np.array([0.25, 0.25, 0.25, 0.25])] * 5
    spread = weight_stability_across_seeds(weights_list)
    assert spread.max() == 0


def test_stability_grows_with_dispersion():
    weights_list = [
        np.array([0.1, 0.2, 0.3, 0.4]),
        np.array([0.4, 0.3, 0.2, 0.1]),
        np.array([0.25, 0.25, 0.25, 0.25]),
    ]
    spread = weight_stability_across_seeds(weights_list)
    assert spread.max() > 0
