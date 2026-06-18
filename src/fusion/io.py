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
    if result.diagnostics is not None:
        payload["diagnostics"] = result.diagnostics
    payload.update(result.metadata)
    Path(path).write_text(json.dumps(payload, indent=2, default=str))


def plot_projection(result: Result, path: str | Path) -> None:
    """Render the SLE distribution with its median and 5–95% credible interval."""
    import matplotlib.pyplot as plt

    from fusion.projection import projection_summary

    if result.projection is None:
        raise ValueError("Result has no projection yet.")
    summary = projection_summary(result.projection)
    fig, ax = plt.subplots()
    ax.hist(result.projection.values, bins=40, color="0.7")
    ax.axvline(
        summary["median"],
        color="C0",
        lw=2,
        label=f"median {summary['median']:.3f} m",
    )
    ax.axvspan(
        summary["lower"],
        summary["upper"],
        color="C0",
        alpha=0.15,
        label=f"{int(summary['lower_q'] * 100)}–{int(summary['upper_q'] * 100)}% CI",
    )
    ax.set_xlabel("SLE 2100 (m)")
    ax.set_ylabel("posterior samples")
    ax.legend()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
