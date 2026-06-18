"""Diagnostic helpers for FUSION runs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr

if TYPE_CHECKING:
    import arviz as az

# Standard NUTS convergence rules of thumb.
RHAT_THRESHOLD = 1.01
ESS_THRESHOLD = 400.0

# The four sampled (non-deterministic) parameters of the v1 model.
_CONVERGENCE_VARS = ("sigma_base_thick", "sigma_base_vel", "beta_thick", "beta_vel")


class ConvergenceWarning(UserWarning):
    """Emitted when post-sampling convergence checks look unhealthy."""


def sampler_diagnostics(
    trace: az.InferenceData,
    var_names: tuple[str, ...] = _CONVERGENCE_VARS,
) -> dict[str, float]:
    """Summarize MCMC sampler health for the sampled parameters.

    Returns the worst-case R-hat and smallest bulk/tail effective sample
    size across ``var_names``, plus the number of divergent transitions.
    ``max_rhat`` > 1.01, low ESS, or any divergence means the posterior
    (and the member weights derived from it) should not be trusted without
    rerunning with more tuning/draws. This is a different question from
    :func:`weight_stability_across_seeds`, which probes subsample sensitivity.
    """
    import arviz as az

    posterior = trace["posterior"]
    present = [v for v in var_names if v in posterior]
    if not present:
        raise ValueError(
            f"None of {var_names} are in the trace posterior; pass var_names "
            "matching your model's sampled parameters."
        )
    # arviz ships types but rhat/ess are themselves unannotated; pin their
    # results to xr.Dataset at this boundary (cf. pm.sample in inference/model).
    rhat: xr.Dataset = az.rhat(trace, var_names=present)  # type: ignore[no-untyped-call]
    ess_bulk: xr.Dataset = az.ess(trace, var_names=present, method="bulk")  # type: ignore[no-untyped-call]
    ess_tail: xr.Dataset = az.ess(trace, var_names=present, method="tail")  # type: ignore[no-untyped-call]
    n_div = 0
    if "sample_stats" in trace.groups() and "diverging" in trace["sample_stats"]:
        n_div = int(trace["sample_stats"]["diverging"].sum())
    return {
        "max_rhat": max(float(rhat[v].max()) for v in present),
        "min_ess_bulk": min(float(ess_bulk[v].min()) for v in present),
        "min_ess_tail": min(float(ess_tail[v].min()) for v in present),
        "n_divergences": float(n_div),
    }


def convergence_warnings(diagnostics: dict[str, float]) -> list[str]:
    """Human-readable warnings for any unhealthy diagnostic. Empty if all good."""
    msgs: list[str] = []
    if diagnostics["n_divergences"] > 0:
        msgs.append(
            f"{int(diagnostics['n_divergences'])} divergent transition(s) during "
            "sampling. The posterior may be biased; raise inference.target_accept "
            "or inference.tune."
        )
    if diagnostics["max_rhat"] > RHAT_THRESHOLD:
        msgs.append(
            f"max R-hat {diagnostics['max_rhat']:.3f} exceeds {RHAT_THRESHOLD}. "
            "Chains have not converged; increase inference.tune/draws."
        )
    if diagnostics["min_ess_bulk"] < ESS_THRESHOLD:
        msgs.append(
            f"min bulk ESS {diagnostics['min_ess_bulk']:.0f} is below "
            f"{ESS_THRESHOLD:.0f}. Effective sample size is low; increase "
            "inference.draws."
        )
    return msgs


def weight_stability_across_seeds(weights_list: list[xr.DataArray]) -> xr.DataArray:
    """Per-member standard deviation across multiple seeded runs.

    A common sanity check: rerun the pipeline with several
    ``inference.subsample.seed`` values and pass the resulting
    per-member weight vectors here. Large σ on a member means the
    weight is sensitive to which 20k pixels happened to be drawn.

    Each input is a ``member``-coorded DataArray (e.g. a column of the
    weights table). Stacking along a new ``seed`` dim aligns runs by
    member label, so the result is correct even if two runs enumerate
    members in different orders.
    """
    # join="outer" aligns runs by member label (and surfaces a member
    # absent from some run as NaN rather than silently dropping it).
    stacked = xr.concat(weights_list, dim="seed", join="outer")
    return stacked.std(dim="seed")
