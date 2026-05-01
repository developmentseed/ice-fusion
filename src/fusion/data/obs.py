"""Observations fetcher with on-disk cache.

Downloads the multi-year obs bundle from Source the first time a given
version is requested, then reads from the local cache on subsequent
runs. The bundle layout (one NetCDF per year per stream) is documented
under ``BUNDLE LAYOUT`` below.

The Source bucket URL below is a placeholder pending Sara's upload
(Apr 27 action item). Once confirmed, replace ``SOURCE_BUCKET_URL`` and
update ``docs/usage.md``.

BUNDLE LAYOUT
-------------
``<source-bucket>/<org>/fusion-obs/<version>/``::

    elevation/
        elev_antarctica_elevation_2015.nc   # vars: height, absolute_elevation_rmse
        elev_antarctica_elevation_2016.nc
        ... (through 2020)
    velocity/
        vel_Antarctica_ice_velocity_2014.nc # vars: VX, VY, ERRX, ERRY
        ... (through 2019)

``load_observations`` returns ``{"elevation": ds, "velocity": ds}`` where
each dataset is the per-year stack with dim ``year`` and coords ``(y, x)``.
"""

import os
import re
from pathlib import Path

import xarray as xr

from fusion.config import ObservationsConfig

SOURCE_BUCKET_URL = "s3://us-west-2.opendata.source.coop/<org>/fusion-obs"
SOURCE_REGION = "us-west-2"

ELEV_VARS = ("height", "absolute_elevation_rmse")
VEL_VARS = ("VX", "VY", "ERRX", "ERRY")


def _cache_dir() -> Path:
    """Read ``FUSION_CACHE`` lazily so callers can set it after import."""
    return Path(os.environ.get("FUSION_CACHE", Path.home() / ".cache" / "fusion"))


def load_observations(cfg: ObservationsConfig) -> dict[str, xr.Dataset]:
    """Return ``{"elevation": ds, "velocity": ds}`` for the requested version.

    Each dataset has dim ``year`` (annual integer) plus ``(y, x)``.
    """
    versioned = _cache_dir() / cfg.version
    if not versioned.exists():
        versioned.mkdir(parents=True, exist_ok=True)
        _fetch_bundle(cfg.version, versioned)
    return {
        "elevation": _stack_yearly(versioned / "elevation", ELEV_VARS),
        "velocity": _stack_yearly(versioned / "velocity", VEL_VARS),
    }


def _stack_yearly(folder: Path, keep_vars: tuple[str, ...]) -> xr.Dataset:
    """Open every NetCDF in ``folder`` and concatenate along a new ``year`` dim.

    The year is extracted from the filename (first 4-digit run). Singleton
    dims (e.g. ``time=1`` on the elevation files) are squeezed before
    concat — they carry per-file values that would otherwise refuse to
    align across years. This matches the prototype's ``_to_2d`` squeeze.
    """
    files = sorted(folder.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No NetCDFs found in {folder}")

    def _preprocess(ds: xr.Dataset) -> xr.Dataset:
        src = ds.encoding.get("source", "")
        m = re.search(r"(\d{4})", Path(src).name)
        if not m:
            raise ValueError(f"No year in filename: {src}")
        return ds[list(keep_vars)].squeeze(drop=True).expand_dims(year=[int(m.group(1))])

    return xr.open_mfdataset(
        files,
        combine="nested",
        concat_dim="year",
        preprocess=_preprocess,
    ).sortby("year")


def _fetch_bundle(version: str, target: Path) -> None:
    """Download the versioned obs bundle from Source via ``obstore``."""
    import obstore
    from obstore.store import from_url

    store = from_url(SOURCE_BUCKET_URL, region=SOURCE_REGION, skip_signature=True)
    for stream in ("elevation", "velocity"):
        out_dir = target / stream
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{version}/{stream}/"
        for batch in obstore.list(store, prefix=prefix):
            for entry in batch:
                key = entry["path"]
                local = out_dir / Path(key).name
                local.write_bytes(bytes(obstore.get(store, key).bytes()))
