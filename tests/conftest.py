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
