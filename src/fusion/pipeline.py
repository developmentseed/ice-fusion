"""Top-level FUSION pipeline.

Each step is callable on its own; ``run`` composes them. The pipeline
shape is ``load_data → prepare → sample → project``; ``prepare``
replaces the earlier ``score`` name (the v1 function flattens inputs
for the PyMC model rather than producing scalar scores).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fusion.config import Config
from fusion.data.ensemble import load_ensemble
from fusion.data.obs import load_observations
from fusion.data.prepare import PreparedData
from fusion.data.prepare import prepare as _prepare_data
from fusion.inference.model import run_inference
from fusion.projection import compute_projection
from fusion.result import Result


def load_data(cfg: Config) -> dict:
    """Fetch observations and load + standardise the ensemble.

    Returns ``{"obs": {...}, "ensemble": <Dataset>}``. Raises
    ``NotImplementedError`` if the ensemble grid does not match the
    obs grid (regridding is a v1.1 feature).
    """
    obs = load_observations(cfg.observations)
    ens = load_ensemble(cfg.ensemble)
    _check_grid_compatible(obs, ens)
    return {"obs": obs, "ensemble": ens}


def prepare(cfg: Config, data: dict) -> PreparedData:
    """Flatten + mask the rate-of-change inputs the PyMC model consumes."""
    return _prepare_data(data["obs"], data["ensemble"], cfg.inference)


def sample(cfg: Config, prepared: PreparedData):
    """Run hierarchical PyMC inference and return the trace."""
    return run_inference(prepared, cfg.inference, random_seed=cfg.inference.seed)


def project(cfg: Config, trace, data: dict):
    """Apply posterior weights to per-member SLE-2100 values."""
    sle = _sle_per_member(data["ensemble"], cfg.projection.target_year)
    # v1 weights are exposed as the deterministic ``w`` in the trace.
    w_post = trace.posterior["w"].stack(sample=("chain", "draw")).values.T
    return compute_projection(sle, w_post)


def run(cfg: Config) -> Result:
    """One-call pipeline: ``load_data → prepare → sample → project``."""
    data = load_data(cfg)
    prepared = prepare(cfg, data)
    trace = sample(cfg, prepared)
    proj = project(cfg, trace, data)
    weights_df = _weights_dataframe(prepared, trace)
    return Result(
        config=cfg,
        data=data,
        prepared=prepared,
        trace=trace,
        weights=weights_df,
        projection=proj,
    )


def _check_grid_compatible(obs: dict, ens) -> None:
    """v1 expects pre-regridded inputs on a shared (y, x) grid.

    Verifies the ``(y, x)`` *shapes* match across obs streams and the
    ensemble. Coordinate *values* are not compared — Sara's reference
    inputs are all 761×761 EPSG:3031 by construction but store the
    coords with different units (m vs km) and dtypes; the v1 metric is
    purely positional so unit-level coord agreement is not required.
    Native-resolution → 8 km regridding (which would require coord-value
    agreement) is a v1.1 feature.
    """
    obs_shape = (obs["elevation"].sizes["y"], obs["elevation"].sizes["x"])
    vel_shape = (obs["velocity"].sizes["y"], obs["velocity"].sizes["x"])
    ens_shape = (ens.sizes["y"], ens.sizes["x"])
    if not (obs_shape == vel_shape == ens_shape):
        raise NotImplementedError(
            "Ensemble grid shape does not match obs grid shape "
            f"(elevation={obs_shape}, velocity={vel_shape}, ensemble={ens_shape}). "
            "v1 requires inputs to be pre-regridded onto a shared 761×761 grid; "
            "native-resolution regridding is a v1.1 feature."
        )


def _sle_per_member(ensemble, target_year):
    """Placeholder SLE-from-h reduction.

    The canonical reduction (volume above flotation × area-equivalent
    SLE conversion) is a Sara-owned open question — see the plan's
    Open implementation questions section. This stub keeps the pipeline
    end-to-end testable but is **not release-quality**.
    """
    return ensemble["h"].mean(dim=("x", "y", "time"))


def _weights_dataframe(prepared: PreparedData, trace) -> pd.DataFrame:
    """Per-member weights table.

    - ``posterior_mean`` / ``posterior_sd`` come from the deterministic
      ``w`` in the PyMC trace.
    - ``point_estimate`` is the prototype's plug-in formula
      (``compute_model_weights`` in ``validation/baseline/full_model.py``):
      compute per-member loglik using posterior-mean ``sigma_base_*``
      and ``beta_*``, normalise by N, softmax.
    - ``point_estimate_loglik`` is the N-scaled per-member Gaussian
      log-likelihood (divided by ``y_obs.size``) at the posterior-mean
      variance — the same number the prototype writes as
      ``log_likelihood`` in ``model_weights_table.csv``. Surfaced so the
      validation harness in ``validation/compare.py`` can diff it
      bit-exactly against the prototype.
    """
    post = trace.posterior["w"]
    mean = post.mean(dim=("chain", "draw")).values
    sd = post.std(dim=("chain", "draw")).values
    point_est, point_est_loglik = plug_in_weights(prepared, trace)
    return pd.DataFrame(
        {
            "member_id": prepared.member_ids,
            "posterior_mean": mean,
            "posterior_sd": sd,
            "point_estimate": point_est,
            "point_estimate_loglik": point_est_loglik,
        }
    )


def plug_in_weights(prepared: PreparedData, trace) -> tuple[np.ndarray, np.ndarray]:
    """Direct port of ``compute_model_weights`` from the prototype.

    Returns ``(weights, loglik)`` where ``weights`` is the
    N-normalised softmax (length M, sums to 1) and ``loglik`` is the
    N-scaled per-member Gaussian log-likelihood (``loglik / y_obs.size``,
    length M), evaluated at the posterior-mean ``sigma_base_*`` /
    ``beta_*``. This second value matches the prototype's
    ``compute_model_weights`` second return and the ``log_likelihood``
    column of ``model_weights_table.csv``. Both are deterministic given
    the trace's posterior means and the prepared arrays, so the
    validation harness uses them as the canonical bit-exact comparison
    points.
    """
    post = trace.posterior
    sigma_base_thick = float(post["sigma_base_thick"].mean(dim=("chain", "draw")).values)
    beta_thick = float(post["beta_thick"].mean(dim=("chain", "draw")).values)
    sigma_base_vel = float(post["sigma_base_vel"].mean(dim=("chain", "draw")).values)
    beta_vel = float(post["beta_vel"].mean(dim=("chain", "draw")).values)

    is_thick = np.arange(prepared.y_obs.size) < prepared.n_dhdt
    sigma_model = np.empty_like(prepared.y_obs, dtype=float)
    sigma_model[is_thick] = sigma_base_thick * np.sqrt(1.0 + beta_thick * prepared.speed[is_thick])
    sigma_model[~is_thick] = sigma_base_vel * np.sqrt(1.0 + beta_vel * prepared.speed[~is_thick])
    sigma_tot = np.sqrt(prepared.sigma_obs**2 + sigma_model**2)

    # Vectorised over the M (member) axis. Per-row sum order preserved
    # vs. the loop version, so this is bit-exact (tests/test_plug_in_weights.py).
    log_var_term = np.log(2 * np.pi * sigma_tot**2)
    R = prepared.y_obs - prepared.F  # broadcasts to (M, N)
    loglik = -0.5 * (R**2 / sigma_tot**2 + log_var_term).sum(axis=1)
    loglik_scaled = loglik / prepared.y_obs.size
    w = np.exp(loglik_scaled - loglik_scaled.max())
    # Return the N-scaled loglik (not the raw sum): this is the prototype's
    # ``compute_model_weights`` second return and the value written to the
    # ``log_likelihood`` column of ``model_weights_table.csv``.
    return w / w.sum(), loglik_scaled
