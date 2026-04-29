from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class EnsembleConfig(BaseModel):
    """Where the ensemble lives on disk and how to read it."""

    path: Path
    adapter: Literal["psuism"]


class ObservationsConfig(BaseModel):
    """Which bundled observations to score against, and where to fetch them."""

    source: Literal["source-coop"]
    version: str


class GridConfig(BaseModel):
    """Target grid that the ensemble is regridded onto before scoring."""

    target: Literal["obs_8km"]
    method: Literal["conservative", "bilinear"] = "conservative"


class RegionsConfig(BaseModel):
    """Region definitions used for per-region scoring."""

    source: Literal["imbie_basins"] = "imbie_basins"


class MetricConfig(BaseModel):
    """Which comparison metric to use. v1 ships only the pixelwise Gaussian likelihood."""

    type: Literal["pixelwise_gaussian"]


class StreamWeights(BaseModel):
    """Per-stream multipliers on the pixelwise log-likelihood.

    ``thick`` weights the thickness-change term; ``vel`` weights the
    velocity-change term.
    """

    thick: float = 1.5
    vel: float = 1.5


class SubsampleConfig(BaseModel):
    """Random subsample applied within each region before the likelihood is summed.

    ``size`` is the maximum number of pixels per region; ``seed`` makes the
    draw deterministic. Use the same ``seed`` as the validation harness in
    ``PSUISM_HBM_V1`` to reproduce the prototype bit-exactly.
    """

    size: int = 20_000
    seed: int = 42


class InferenceConfig(BaseModel):
    """PyMC inference settings.

    Includes weights on the per-stream likelihood, the subsample, and the
    standard NUTS knobs (``draws``, ``tune``, ``chains``, ``target_accept``).
    """

    stream_weights: StreamWeights = Field(default_factory=StreamWeights)
    subsample: SubsampleConfig = Field(default_factory=SubsampleConfig)
    draws: int = 500
    tune: int = 1000
    chains: int = 4
    target_accept: float = 0.95


class ProjectionConfig(BaseModel):
    """Which forward-projection target to compute when applying the posterior weights.

    Specifies the target year and the quantity to project.
    """

    target_year: int = 2100
    quantity: Literal["grounded_ice_volume"] = "grounded_ice_volume"


class Config(BaseModel):
    """Top-level FUSION run configuration. Loaded from a YAML file via ``load_config``."""

    ensemble: EnsembleConfig
    observations: ObservationsConfig
    grid: GridConfig
    regions: RegionsConfig = Field(default_factory=RegionsConfig)
    metric: MetricConfig
    inference: InferenceConfig
    projection: ProjectionConfig


def load_config(path: str | Path) -> Config:
    """Load and validate a FUSION config from a YAML file.

    Validation errors (and other read errors) are raised as ``ValueError``
    with the path included in the message.
    """
    raw = yaml.safe_load(Path(path).read_text())
    try:
        return Config.model_validate(raw)
    except Exception as e:
        raise ValueError(f"Invalid config at {path}: {e}") from e
