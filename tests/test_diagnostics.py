import arviz as az
import numpy as np
import pytest
import xarray as xr

from fusion.diagnostics import (
    ConvergenceWarning,
    convergence_warnings,
    sampler_diagnostics,
    weight_stability_across_seeds,
)

_VARS = ("sigma_base_thick", "sigma_base_vel", "beta_thick", "beta_vel")


def _healthy_trace(seed: int = 0) -> az.InferenceData:
    """4 well-mixed chains × 1000 draws → R-hat ≈ 1, high ESS, no divergences."""
    rng = np.random.default_rng(seed)
    post = {v: rng.normal(0.5, 0.1, (4, 1000)) for v in _VARS}
    stats = {"diverging": np.zeros((4, 1000), dtype=bool)}
    trace: az.InferenceData = az.from_dict(posterior=post, sample_stats=stats)
    return trace


def _unconverged_trace(seed: int = 2) -> az.InferenceData:
    """2 chains around very different means → R-hat ≫ 1 (finite, nonzero variance)."""
    rng = np.random.default_rng(seed)
    c0 = rng.normal(0.0, 0.01, (1, 500))
    c1 = rng.normal(10.0, 0.01, (1, 500))
    post = {v: np.concatenate([c0, c1], axis=0) for v in _VARS}
    stats = {"diverging": np.zeros((2, 500), dtype=bool)}
    trace: az.InferenceData = az.from_dict(posterior=post, sample_stats=stats)
    return trace


def _divergent_trace(seed: int = 1) -> az.InferenceData:
    """Healthy mixing but 4 divergent transitions flagged."""
    rng = np.random.default_rng(seed)
    post = {v: rng.normal(0.5, 0.1, (4, 1000)) for v in _VARS}
    diverging = np.zeros((4, 1000), dtype=bool)
    diverging[:, 0] = True  # one per chain → 4 total
    trace: az.InferenceData = az.from_dict(posterior=post, sample_stats={"diverging": diverging})
    return trace


def test_sampler_diagnostics_reports_healthy_trace() -> None:
    diag = sampler_diagnostics(_healthy_trace())
    assert set(diag) == {"max_rhat", "min_ess_bulk", "min_ess_tail", "n_divergences"}
    assert diag["max_rhat"] < 1.01
    assert diag["min_ess_bulk"] > 400
    assert diag["n_divergences"] == 0
    assert convergence_warnings(diag) == []


def test_sampler_diagnostics_flags_nonconvergence() -> None:
    diag = sampler_diagnostics(_unconverged_trace())
    assert diag["max_rhat"] > 1.01
    assert any("R-hat" in m for m in convergence_warnings(diag))


def test_sampler_diagnostics_counts_divergences() -> None:
    diag = sampler_diagnostics(_divergent_trace())
    assert diag["n_divergences"] == 4
    assert any("divergent" in m for m in convergence_warnings(diag))


def test_sampler_diagnostics_raises_on_missing_vars() -> None:
    with pytest.raises(ValueError):
        sampler_diagnostics(_healthy_trace(), var_names=("nonexistent",))


def test_sampler_diagnostics_on_real_trace() -> None:
    """Exercise the real arviz path on an actual short PyMC trace."""
    from fusion.config import InferenceConfig
    from fusion.inference.model import run_inference
    from tests.test_inference import _toy_prepared

    cfg = InferenceConfig(draws=40, tune=40, chains=2)
    trace = run_inference(_toy_prepared(), cfg, random_seed=0, progressbar=False)
    diag = sampler_diagnostics(trace)
    assert diag["n_divergences"] >= 0
    assert np.isfinite(diag["max_rhat"])


def test_convergence_warning_is_a_userwarning() -> None:
    assert issubclass(ConvergenceWarning, UserWarning)


def _weights(values: list[float], members: list[str]) -> xr.DataArray:
    return xr.DataArray(values, dims=["member"], coords={"member": members})


def test_stability_across_identical_inputs_is_zero() -> None:
    members = list("ABCD")
    weights_list = [_weights([0.25, 0.25, 0.25, 0.25], members) for _ in range(5)]
    spread = weight_stability_across_seeds(weights_list)
    assert float(spread.max()) == 0


def test_stability_grows_with_dispersion() -> None:
    members = list("ABCD")
    weights_list = [
        _weights([0.1, 0.2, 0.3, 0.4], members),
        _weights([0.4, 0.3, 0.2, 0.1], members),
        _weights([0.25, 0.25, 0.25, 0.25], members),
    ]
    spread = weight_stability_across_seeds(weights_list)
    assert float(spread.max()) > 0


def test_stability_aligns_members_by_label() -> None:
    """Runs with members in different orders must align by label, not position."""
    a = _weights([0.1, 0.2, 0.3, 0.4], list("ABCD"))
    # Same per-member values, members listed in reverse order.
    b = _weights([0.4, 0.3, 0.2, 0.1], list("DCBA"))
    spread = weight_stability_across_seeds([a, b])
    # Each member has identical values across the two runs → zero spread.
    assert float(spread.max()) == 0
    assert list(spread["member"].values) == list("ABCD") or set(spread["member"].values) == set(
        "ABCD"
    )
