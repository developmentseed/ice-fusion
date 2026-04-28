"""Persistence helpers for FUSION run outputs."""
from pathlib import Path

from fusion.result import Result


def save_weights(result: Result, path: str | Path) -> None:
    """Write the weights DataFrame to CSV."""
    raise NotImplementedError("Implemented in Task 13.")


def save_metadata(result: Result, path: str | Path) -> None:
    """Write run_metadata.json with full reproducibility info."""
    raise NotImplementedError("Implemented in Task 13.")


def plot_projection(result: Result, path: str | Path) -> None:
    """Render the SLE-2100 distribution plot."""
    raise NotImplementedError("Implemented in Task 14.")
