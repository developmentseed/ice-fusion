from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class EnsembleConfig(BaseModel):
    path: Path
    adapter: Literal["psuism"]


class ObservationsConfig(BaseModel):
    source: Literal["source-coop"]
    version: str


class GridConfig(BaseModel):
    target: Literal["obs_8km"]
    method: Literal["conservative", "bilinear"] = "conservative"


class RegionsConfig(BaseModel):
    source: Literal["imbie_basins"] = "imbie_basins"


class MetricConfig(BaseModel):
    type: Literal["pixelwise_gaussian"]


class StreamWeights(BaseModel):
    thick: float = 1.5
    vel: float = 1.5


class SubsampleConfig(BaseModel):
    size: int = 20_000
    seed: int = 42


class InferenceConfig(BaseModel):
    stream_weights: StreamWeights = Field(default_factory=StreamWeights)
    subsample: SubsampleConfig = Field(default_factory=SubsampleConfig)
    draws: int = 500
    tune: int = 1000
    chains: int = 4
    target_accept: float = 0.95


class ProjectionConfig(BaseModel):
    target_year: int = 2100
    quantity: Literal["grounded_ice_volume"] = "grounded_ice_volume"


class Config(BaseModel):
    ensemble: EnsembleConfig
    observations: ObservationsConfig
    grid: GridConfig
    regions: RegionsConfig = Field(default_factory=RegionsConfig)
    metric: MetricConfig
    inference: InferenceConfig
    projection: ProjectionConfig


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    try:
        return Config.model_validate(raw)
    except Exception as e:
        raise ValueError(f"Invalid config at {path}: {e}") from e
