"""Forward projection: weight per-member SLE values by the posterior weights.

The mapping ``member → SLE-2100`` is **not defined by the prototype** —
the prototype produces weights only. ``compute_projection`` ships
agnostic to that mapping: callers pass the per-member SLE values they've
already reduced (volume above flotation × area-equivalent SLE
conversion, or whatever the science owner specifies). The pipeline supplies a
placeholder reduction (``ensemble["h"].mean(("x","y"))``) that is
acceptable for end-to-end smoke tests but **not for the v1 release** —
see plan §open implementation questions.
"""

from __future__ import annotations

import xarray as xr


def compute_projection(
    sle_per_member: xr.DataArray,
    weights_posterior: xr.DataArray,
) -> xr.DataArray:
    """Combine per-member SLE values with posterior weight samples.

    Both inputs carry a ``member`` coordinate, so the contraction is
    aligned by member label rather than by array position — if the
    weight and SLE member orders ever diverged, this would still produce
    the correct per-member pairing (and raise on a genuine mismatch)
    instead of silently transposing weights onto the wrong members.

    Parameters
    ----------
    sle_per_member
        Per-member SLE-2100 values, dim ``member``.
    weights_posterior
        Posterior weight samples with dims ``(sample, member)`` (any
        sample-like dim name works; only ``member`` is contracted).

    Returns
    -------
    xr.DataArray
        Weighted SLE distribution over the surviving sample dim.
    """
    # xr.dot is untyped (returns Any); pin the type at this boundary.
    projection: xr.DataArray = xr.dot(weights_posterior, sle_per_member, dim="member")
    return projection
