"""Container for FUSION run outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    import arviz as az

    from fusion.config import Config
    from fusion.data.prepare import PreparedData
    from fusion.pipeline import LoadedData


@dataclass
class Result:
    """Outputs of a FUSION run.

    Held together as a single object so callers can pass intermediate
    state between steps without juggling multiple variables.
    """

    config: Config | None = None
    data: LoadedData | None = None
    prepared: PreparedData | None = None
    trace: az.InferenceData | None = None
    weights: pd.DataFrame | None = None
    projection: xr.DataArray | None = None
    diagnostics: dict[str, float] | None = None
    metadata: dict[str, object] = field(default_factory=dict)
