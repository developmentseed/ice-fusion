"""Container for FUSION run outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from fusion.data.prepare import PreparedData


@dataclass
class Result:
    """Outputs of a FUSION run.

    Held together as a single object so callers can pass intermediate
    state between steps without juggling multiple variables.
    """

    config: Any
    data: dict[str, xr.Dataset] = field(default_factory=dict)
    prepared: PreparedData | None = None
    trace: Any = None
    weights: pd.DataFrame | None = None
    projection: xr.Dataset | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
