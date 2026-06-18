"""FUSION — score ice-sheet ensembles against observations; weight projections."""

from fusion._version import __version__
from fusion.config import Config, load_config
from fusion.diagnostics import (
    ConvergenceWarning,
    sampler_diagnostics,
    weight_stability_across_seeds,
)
from fusion.io import plot_projection, save_metadata, save_weights
from fusion.pipeline import load_data, prepare, project, run, sample
from fusion.result import Result

__all__ = [
    "__version__",
    "Config",
    "Result",
    "load_config",
    "load_data",
    "prepare",
    "sample",
    "project",
    "run",
    "save_weights",
    "save_metadata",
    "plot_projection",
    "weight_stability_across_seeds",
    "sampler_diagnostics",
    "ConvergenceWarning",
]
