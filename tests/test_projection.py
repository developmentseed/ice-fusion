import numpy as np
import xarray as xr

from fusion.projection import compute_projection


def test_projection_weighted_distribution() -> None:
    """SLE distribution should be a per-sample weighted sum of member SLEs."""
    sle_per_member = xr.DataArray(
        [0.10, 0.20, 0.30, 0.40],
        dims=["member"],
        coords={"member": list("ABCD")},
    )
    weights_post = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.4, 0.3, 0.2, 0.1],
        ]
    )
    proj = compute_projection(sle_per_member, weights_post)
    assert proj.sizes["sample"] == 2
    expected_0 = 0.1 * 0.1 + 0.2 * 0.2 + 0.3 * 0.3 + 0.4 * 0.4
    expected_1 = 0.4 * 0.1 + 0.3 * 0.2 + 0.2 * 0.3 + 0.1 * 0.4
    np.testing.assert_allclose(proj.values, [expected_0, expected_1])
