import numpy as np
import xarray as xr

from fusion.data.time_utils import (
    _clean_and_mask_time,
    model_decimal_years_from_ds,
    snap_model_year_to_obs_year,
)


def test_decimal_years_decode_seconds_since() -> None:
    t = np.array([0.0, 365.2425 * 86400.0, 1e30], dtype=np.float64)
    ds = xr.Dataset(
        coords={"time": ("time", t)},
    )
    ds["time"].attrs = {"units": "seconds since 2000-01-01 00:00:00"}
    yrs = model_decimal_years_from_ds(ds)
    assert yrs.shape == (2,)  # the 1e30 row is dropped
    np.testing.assert_allclose(yrs, [2000.0, 2001.0], rtol=1e-3)


def test_decimal_years_passthrough_when_already_years() -> None:
    t = np.array([2010.0, 2011.0, 2012.0])
    ds = xr.Dataset(coords={"time": ("time", t)})
    yrs = model_decimal_years_from_ds(ds)
    np.testing.assert_array_equal(yrs, t)


def test_clean_and_mask_drops_spurious_rows() -> None:
    t = np.array([0.0, 1e30, 31_557_600.0])  # second row is bogus
    ds = xr.Dataset(
        data_vars={"h": (("time",), np.array([1.0, 2.0, 3.0]))},
        coords={"time": ("time", t)},
    )
    cleaned = _clean_and_mask_time(ds)
    assert cleaned.sizes["time"] == 2
    np.testing.assert_array_equal(cleaned["h"].values, [1.0, 3.0])


def test_year_snap_within_tol() -> None:
    available = np.array([2014, 2015, 2016])
    assert snap_model_year_to_obs_year(2015.4, available) == 2015
    assert snap_model_year_to_obs_year(2014.0, available) == 2014


def test_year_snap_returns_none_outside_tol() -> None:
    available = np.array([2014, 2015, 2016])
    assert snap_model_year_to_obs_year(2018.0, available) is None
