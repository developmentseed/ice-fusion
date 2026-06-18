"""Time-axis helpers for the model side of the comparison.

Ported from the prototype (``validation/baseline/full_model.py``):
``model_decimal_years_from_ds``, ``_parse_origin_allow_day00``,
``_clean_and_mask_time``, and ``snap_model_year_to_obs_year`` (renamed
from the prototype's ``_snap_model_year_to_obs_year`` so the public
helper has no leading underscore).

These live in ``data/`` rather than the ensemble adapter because the
obs side needs the same year-snap helper to align observation years
with model interval endpoints.
"""

import calendar
import re
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

from fusion._array_types import BoolArray, FloatArray, IntArray

SEC_PER_YEAR = 365.2425 * 86400.0


def _parse_origin_allow_day00(unit_str: str) -> datetime:
    """Parse ``"seconds since YYYY-MM-DD hh:mm:ss"`` but allow ``DD=00``.

    Some PSU-ISM files use ``DD=00`` to mean "the last day of the previous
    month" — handle that without raising.
    """
    m = re.search(r"since\s+(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})", unit_str)
    if not m:
        raise ValueError(f"Could not parse time unit string: {unit_str}")

    Y, Mo, D, hh, mm, ss = map(int, m.groups())

    if D == 0:
        if Mo == 1:
            Y2, Mo2 = Y - 1, 12
        else:
            Y2, Mo2 = Y, Mo - 1
        last_day = calendar.monthrange(Y2, Mo2)[1]
        return datetime(Y2, Mo2, last_day, hh, mm, ss)

    return datetime(Y, Mo, D, hh, mm, ss)


def _finite_time_mask(time_var: xr.DataArray) -> tuple[FloatArray, BoolArray]:
    """Return ``(t_cleaned, good_mask)`` for a time DataArray.

    Replaces ``_FillValue`` / ``missing_value`` / ``|t| > 1e20`` placeholder
    rows with NaN and returns the boolean mask of finite entries alongside
    the cleaned float array.
    """
    t = np.array(time_var.values, dtype=float, copy=True)
    fill_value = time_var.attrs.get("_FillValue")
    missing_value = time_var.attrs.get("missing_value")
    if fill_value is not None:
        t[t == fill_value] = np.nan
    if missing_value is not None:
        t[t == missing_value] = np.nan
    t[np.abs(t) > 1e20] = np.nan
    good = np.isfinite(t)
    if not np.any(good):
        raise ValueError("No valid time values found in time variable")
    return t, good


def model_decimal_years_from_ds(ds: xr.Dataset, time_name: str = "time") -> FloatArray:
    """Convert a model dataset's time axis to decimal years.

    Handles:
    - ``seconds since YYYY-MM-DD hh:mm:ss`` units (decoded → decimal year).
    - Already-decimal-year values (returned as-is).
    - Missing values (``_FillValue`` / ``missing_value`` / ``|t| > 1e20``).

    Returns the cleaned 1-D array — invalid entries are removed.
    """
    time_var = ds[time_name]
    t_all, good = _finite_time_mask(time_var)
    t: FloatArray = t_all[good]

    unit_str = time_var.attrs.get("units") or time_var.attrs.get("unit")
    if unit_str is None or "since" not in unit_str or "second" not in unit_str:
        # Already-decimal-year values, or units we don't understand — assume years.
        return t

    origin = _parse_origin_allow_day00(unit_str)
    years = np.empty_like(t, dtype=float)
    for i, sec in enumerate(t):
        dt = origin + timedelta(seconds=float(sec))
        year_start = datetime(dt.year, 1, 1)
        frac = (dt - year_start).total_seconds() / SEC_PER_YEAR
        years[i] = dt.year + frac
    return years


def _clean_and_mask_time(ds: xr.Dataset, time_name: str = "time") -> xr.Dataset:
    """Drop time-axis rows whose time value is missing or absurd.

    PSU-ISM files include a spurious second time entry (placeholder values
    around ``1e30``). This helper masks those out before any further
    processing so subsequent steps see only real time steps.
    """
    _, good = _finite_time_mask(ds[time_name])
    return ds.isel({time_name: good})


def snap_model_year_to_obs_year(
    t_model: float,
    available_years: IntArray,
    tol: float = 1.5,
) -> int | None:
    """Map a model decimal year to the nearest available obs integer year.

    Returns ``None`` if no obs year is within ``tol`` years of ``t_model``.
    """
    available_years = np.asarray(available_years)
    idx = int(np.argmin(np.abs(available_years - t_model)))
    if abs(available_years[idx] - t_model) <= tol:
        return int(available_years[idx])
    return None
