import numpy as np
import pytest
import xarray as xr

from fusion.projection import compute_projection, projection_summary


def test_projection_summary_basic() -> None:
    proj = xr.DataArray(np.arange(0.0, 101.0), dims=["sample"])  # 0..100 inclusive
    s = projection_summary(proj)
    assert s["median"] == 50.0
    assert s["lower"] == pytest.approx(5.0)
    assert s["upper"] == pytest.approx(95.0)
    assert s["lower_q"] == 0.05 and s["upper_q"] == 0.95
    assert s["sd"] > 0


def test_projection_summary_custom_interval() -> None:
    proj = xr.DataArray(np.arange(0.0, 101.0), dims=["sample"])
    s = projection_summary(proj, lower_q=0.25, upper_q=0.75)
    assert s["lower"] == pytest.approx(25.0)
    assert s["upper"] == pytest.approx(75.0)


def test_projection_weighted_distribution() -> None:
    """SLE distribution should be a per-sample weighted sum of member SLEs."""
    sle_per_member = xr.DataArray(
        [0.10, 0.20, 0.30, 0.40],
        dims=["member"],
        coords={"member": list("ABCD")},
    )
    weights_post = xr.DataArray(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.4, 0.3, 0.2, 0.1],
        ],
        dims=["sample", "member"],
        coords={"member": list("ABCD"), "sample": [0, 1]},
    )
    proj = compute_projection(sle_per_member, weights_post)
    assert proj.sizes["sample"] == 2
    expected_0 = 0.1 * 0.1 + 0.2 * 0.2 + 0.3 * 0.3 + 0.4 * 0.4
    expected_1 = 0.4 * 0.1 + 0.3 * 0.2 + 0.2 * 0.3 + 0.1 * 0.4
    np.testing.assert_allclose(proj.values, [expected_0, expected_1])


def test_projection_aligns_members_by_label() -> None:
    """Weights and SLE in different member orders must align by label, not position."""
    sle_per_member = xr.DataArray(
        [0.10, 0.20, 0.30, 0.40],
        dims=["member"],
        coords={"member": list("ABCD")},
    )
    # Same weights as test above, sample 0, but members listed in reverse.
    weights_post = xr.DataArray(
        [[0.4, 0.3, 0.2, 0.1]],
        dims=["sample", "member"],
        coords={"member": list("DCBA"), "sample": [0]},
    )
    proj = compute_projection(sle_per_member, weights_post)
    # Aligned by label: A→0.1·0.1, B→0.2·0.2, C→0.3·0.3, D→0.4·0.4.
    expected = 0.1 * 0.1 + 0.2 * 0.2 + 0.3 * 0.3 + 0.4 * 0.4
    np.testing.assert_allclose(proj.values, [expected])
