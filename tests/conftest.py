from pathlib import Path

import numpy as np
import pytest
import xarray as xr

FIXTURES = Path(__file__).parent / "data" / "fixtures"


@pytest.fixture(scope="session")
def synthetic_obs_bundle(tmp_path_factory) -> Path:
    """Tiny multi-year obs bundle laid out as Source would host it.

    ``elevation/`` has one NetCDF per year (2015–2020) with ``height`` and
    ``absolute_elevation_rmse``; ``velocity/`` has one per year (2014–2019)
    with ``VX``, ``VY``, ``ERRX``, ``ERRY``. Variable names match the prototype.
    """
    base = tmp_path_factory.mktemp("obs_bundle")
    x = np.arange(-3040000, -2960000, 8000)  # 10 cells
    y = np.arange(-1080000, -1000000, 8000)  # 10 cells
    rng = np.random.default_rng(0)

    elev_dir = base / "elevation"
    elev_dir.mkdir()
    for year in range(2015, 2021):
        ds = xr.Dataset(
            data_vars={
                "height": (("y", "x"), rng.normal(2000, 50, (10, 10))),
                "absolute_elevation_rmse": (("y", "x"), np.full((10, 10), 5.0)),
            },
            coords={"x": x, "y": y},
            attrs={"year": year},
        )
        ds.to_netcdf(elev_dir / f"elev_antarctica_elevation_{year}.nc")

    vel_dir = base / "velocity"
    vel_dir.mkdir()
    for year in range(2014, 2020):
        ds = xr.Dataset(
            data_vars={
                "VX": (("y", "x"), rng.normal(50, 10, (10, 10))),
                "VY": (("y", "x"), rng.normal(-30, 8, (10, 10))),
                "ERRX": (("y", "x"), np.full((10, 10), 2.0)),
                "ERRY": (("y", "x"), np.full((10, 10), 2.0)),
            },
            coords={"x": x, "y": y},
            attrs={"year": year},
        )
        ds.to_netcdf(vel_dir / f"vel_Antarctica_ice_velocity_{year}.nc")
    return base


@pytest.fixture(scope="session")
def synthetic_psuism_dir(tmp_path_factory) -> Path:
    """Two synthetic PSU-ISM members with the prototype's quirks.

    Each file has:
    - ``time`` with units ``"seconds since 1950-01-01 00:00:00"``.
    - A spurious second time row (value ``1e30``) that the adapter must drop.
    - Variables ``h``, ``ua``, ``va`` on ``(time, y, x)``.

    The (cleaned) years span 2014–2020, overlapping both elev (2015–2020)
    and velocity (2014–2019) obs windows so downstream tests of
    year-snapping have plenty of common years to work with.
    """
    base = tmp_path_factory.mktemp("psuism")
    rng = np.random.default_rng(42)
    x = np.arange(-3040000, -2960000, 8000)  # 10 cells
    y = np.arange(-1080000, -1000000, 8000)  # 10 cells

    real_years = [2014, 2015, 2016, 2017, 2018, 2019, 2020]
    seconds = np.array(
        [(yr - 1950) * 365.2425 * 86400.0 for yr in real_years],
        dtype=np.float64,
    )
    bogus = np.array([1e30], dtype=np.float64)
    t = np.concatenate([seconds[:1], bogus, seconds[1:]])
    n = len(t)

    for i, member in enumerate(
        ["regriddedrun01_fort.92_regridded", "regriddedrun07_fort.92_regridded"]
    ):
        ds = xr.Dataset(
            data_vars={
                "h": (("time", "y", "x"), rng.normal(2000 + 50 * i, 80, (n, 10, 10))),
                "ua": (("time", "y", "x"), rng.normal(50, 12, (n, 10, 10))),
                "va": (("time", "y", "x"), rng.normal(-30, 10, (n, 10, 10))),
            },
            coords={"x": x, "y": y, "time": t},
        )
        ds["time"].attrs = {"units": "seconds since 1950-01-01 00:00:00"}
        ds.to_netcdf(base / f"{member}.nc")
    return base
