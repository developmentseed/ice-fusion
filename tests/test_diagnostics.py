import xarray as xr

from fusion.diagnostics import weight_stability_across_seeds


def _weights(values: list[float], members: list[str]) -> xr.DataArray:
    return xr.DataArray(values, dims=["member"], coords={"member": members})


def test_stability_across_identical_inputs_is_zero() -> None:
    members = list("ABCD")
    weights_list = [_weights([0.25, 0.25, 0.25, 0.25], members) for _ in range(5)]
    spread = weight_stability_across_seeds(weights_list)
    assert float(spread.max()) == 0


def test_stability_grows_with_dispersion() -> None:
    members = list("ABCD")
    weights_list = [
        _weights([0.1, 0.2, 0.3, 0.4], members),
        _weights([0.4, 0.3, 0.2, 0.1], members),
        _weights([0.25, 0.25, 0.25, 0.25], members),
    ]
    spread = weight_stability_across_seeds(weights_list)
    assert float(spread.max()) > 0


def test_stability_aligns_members_by_label() -> None:
    """Runs with members in different orders must align by label, not position."""
    a = _weights([0.1, 0.2, 0.3, 0.4], list("ABCD"))
    # Same per-member values, members listed in reverse order.
    b = _weights([0.4, 0.3, 0.2, 0.1], list("DCBA"))
    spread = weight_stability_across_seeds([a, b])
    # Each member has identical values across the two runs → zero spread.
    assert float(spread.max()) == 0
    assert list(spread["member"].values) == list("ABCD") or set(spread["member"].values) == set(
        "ABCD"
    )
