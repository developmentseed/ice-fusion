"""Diagnostic helpers for FUSION runs."""

from __future__ import annotations

import xarray as xr


def weight_stability_across_seeds(weights_list: list[xr.DataArray]) -> xr.DataArray:
    """Per-member standard deviation across multiple seeded runs.

    A common sanity check: rerun the pipeline with several
    ``inference.subsample.seed`` values and pass the resulting
    per-member weight vectors here. Large σ on a member means the
    weight is sensitive to which 20k pixels happened to be drawn.

    Each input is a ``member``-coorded DataArray (e.g. a column of the
    weights table). Stacking along a new ``seed`` dim aligns runs by
    member label, so the result is correct even if two runs enumerate
    members in different orders.
    """
    # join="outer" aligns runs by member label (and surfaces a member
    # absent from some run as NaN rather than silently dropping it).
    stacked = xr.concat(weights_list, dim="seed", join="outer")
    return stacked.std(dim="seed")
