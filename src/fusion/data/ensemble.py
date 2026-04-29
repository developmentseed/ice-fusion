"""PSU-ISM ensemble adapter.

Reads ``*.nc`` files from a directory and emits an xarray Dataset with
dims ``(member, time, y, x)`` and variables ``h, ua, va`` (matching the
prototype). Per-member time cleaning runs before stacking — different
members may have different valid time vectors, so cleaning happens
file-by-file.

Variable names are deliberately not renamed here; the v1 metric and the
validation harness both reference ``h``, ``ua``, ``va`` directly.
"""

import re
from pathlib import Path

import xarray as xr

from fusion.config import EnsembleConfig
from fusion.data.time_utils import _clean_and_mask_time, model_decimal_years_from_ds


def load_ensemble(cfg: EnsembleConfig) -> xr.Dataset:
    """Dispatch to the configured adapter and return a canonical Dataset."""
    if cfg.adapter == "psuism":
        return _load_psuism(cfg.path)
    raise ValueError(f"Unknown adapter: {cfg.adapter}")


def _load_psuism(root: Path) -> xr.Dataset:
    files = sorted(Path(root).glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No PSU-ISM NetCDFs found in {root}")
    members = []
    for f in files:
        ds = xr.open_dataset(f, decode_times=False)
        ds = _clean_and_mask_time(ds, time_name="time")
        years = model_decimal_years_from_ds(ds, time_name="time")
        ds = ds.assign_coords(time=("time", years))
        ds = ds.expand_dims(member=[_member_id(f)])
        members.append(ds)
    return xr.concat(members, dim="member", join="outer")


def _member_id(path: Path) -> str:
    """Extract a short member id from the PSU-ISM filename.

    ``regriddedrun07_fort.92_regridded.nc`` → ``run07``. Falls back to
    the file stem if no ``runNN`` token is present.
    """
    m = re.search(r"run\d+", path.name)
    return m.group(0) if m else path.stem
