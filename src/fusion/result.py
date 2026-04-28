from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


@dataclass
class Result:
    """Outputs of a FUSION run.

    Held together as a single object so callers can pass intermediate
    state between steps without juggling multiple variables.
    """
    config: Any
    data: dict[str, xr.Dataset] = field(default_factory=dict)
    scores: np.ndarray | None = None
    trace: Any = None
    weights: pd.DataFrame | None = None
    projection: xr.Dataset | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
