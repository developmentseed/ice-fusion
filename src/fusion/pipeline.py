"""Top-level FUSION pipeline.

Each step is callable on its own; `run` composes them.
"""
from fusion.config import Config
from fusion.result import Result


def load_data(cfg: Config) -> dict:
    """Fetch observations, load and standardise the ensemble, regrid to 8 km."""
    raise NotImplementedError("Implemented in Task 12; see plan.")


def score(cfg: Config, data: dict):
    """Compute the M × R log-likelihood matrix from data + metric."""
    raise NotImplementedError("Implemented in Task 12; see plan.")


def sample(cfg: Config, scores):
    """Run PyMC inference and return the trace."""
    raise NotImplementedError("Implemented in Task 12; see plan.")


def project(cfg: Config, trace, data: dict):
    """Apply weights to forward runs to get the SLE-2100 distribution."""
    raise NotImplementedError("Implemented in Task 12; see plan.")


def run(cfg: Config) -> Result:
    """One-call pipeline: load_data → score → sample → project."""
    raise NotImplementedError("Implemented in Task 12; see plan.")
