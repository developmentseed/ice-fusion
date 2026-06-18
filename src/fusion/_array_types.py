"""Shared numpy array type aliases.

The rate-of-change pipeline deliberately mixes precisions: model/obs rate
arrays are kept at ``float32`` (a storage-size optimisation that is also
load-bearing for bit-exactness against the prototype), while the arrays the
PyMC model finally consumes are promoted to ``float64``. The aliases below name
both so annotations stay precise about which precision a function traffics in,
rather than collapsing everything to a bare ``np.ndarray`` (which mypy reads as
``ndarray[Any, dtype[Any]]``).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

#: Compute-precision arrays the PyMC model consumes.
FloatArray = npt.NDArray[np.float64]

#: Storage-precision rate arrays produced in ``data.prepare``.
Float32Array = npt.NDArray[np.float32]

#: Index / count arrays (e.g. subsample draws).
IntArray = npt.NDArray[np.intp]

#: Boolean masks.
BoolArray = npt.NDArray[np.bool_]
