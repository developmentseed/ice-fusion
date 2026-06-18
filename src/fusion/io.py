"""Persistence helpers for FUSION run outputs."""

from __future__ import annotations

import json
from pathlib import Path

from fusion._version import __version__
from fusion.metadata import environment_info
from fusion.result import Result


def save_weights(result: Result, path: str | Path) -> None:
    """Write the weights DataFrame to CSV."""
    if result.weights is None:
        raise ValueError("Result has no weights yet — did you forget to run the pipeline?")
    result.weights.to_csv(path, index=False)


def save_metadata(result: Result, path: str | Path) -> None:
    """Write ``run_metadata.json`` with full reproducibility info.

    Includes the resolved ``Config`` (so the validation harness can read
    off ``observations.version``, ``inference.subsample.seed``, the
    stream weights, etc.) alongside the ``fusion`` version and a minimal
    environment fingerprint. Anything in ``result.metadata`` is merged
    in last so callers can attach extra fields (e.g. file hashes).
    """
    payload: dict[str, object] = {
        "fusion_version": __version__,
        "environment": environment_info(),
    }
    if result.config is not None:
        # ``Config`` is a pydantic model in v1; dump it serialisably.
        if hasattr(result.config, "model_dump"):
            payload["config"] = result.config.model_dump(mode="json")
        else:
            payload["config"] = result.config
    payload.update(result.metadata)
    Path(path).write_text(json.dumps(payload, indent=2, default=str))


def plot_projection(result: Result, path: str | Path) -> None:
    """Render the SLE-2100 distribution as a histogram."""
    import matplotlib.pyplot as plt

    if result.projection is None:
        raise ValueError("Result has no projection yet.")
    fig, ax = plt.subplots()
    ax.hist(result.projection.values, bins=40)
    ax.set_xlabel("SLE 2100 (m)")
    ax.set_ylabel("posterior samples")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
