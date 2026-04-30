"""Forward projection: weight per-member SLE values by the posterior weights.

The mapping ``member → SLE-2100`` is **not defined by the prototype** —
the prototype produces weights only. ``compute_projection`` ships
agnostic to that mapping: callers pass the per-member SLE values they've
already reduced (volume above flotation × area-equivalent SLE
conversion, or whatever Sara specifies). The pipeline supplies a
placeholder reduction (``ensemble["h"].mean(("x","y"))``) that is
acceptable for end-to-end smoke tests but **not for the v1 release** —
see plan §open implementation questions.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def compute_projection(
    sle_per_member: xr.DataArray,
    weights_posterior: np.ndarray,
) -> xr.DataArray:
    """Combine per-member SLE values with posterior weight samples.

    Parameters
    ----------
    sle_per_member
        Per-member SLE-2100 values, shape ``(n_members,)`` with a
        ``member`` coord.
    weights_posterior
        Posterior weight samples, shape ``(n_samples, n_members)``.

    Returns
    -------
    xr.DataArray
        Weighted SLE distribution, shape ``(sample,)``.
    """
    return xr.DataArray(
        weights_posterior @ sle_per_member.values,
        dims=["sample"],
        coords={"sample": np.arange(weights_posterior.shape[0])},
    )
