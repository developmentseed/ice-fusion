"""Hierarchical Bayesian model over ensemble member weights. **Sara's territory.**

Direct port of ``build_model_proposal`` and ``run_mcmc`` from the canonical
prototype (pinned at ``validation/baseline/full_model.py``;
<https://github.com/sc-peters/PSUISM_HBM_V1/blob/main/push%204_13_26/full_model.py>).

Priors:
- ``sigma_base_thick ~ HalfNormal(0.5)``, ``sigma_base_vel ~ HalfNormal(0.6)``
- ``beta_thick ~ HalfNormal(0.1)``,  ``beta_vel ~ HalfNormal(0.1)``

Per-pixel uncertainty is inflated by the speed-dependent factor
``sigma_base * sqrt(1 + beta * speed)``; the velocity stream gets an
additional fixed ``+5²`` inflation to prevent it from dominating.

Per-member, per-stream Gaussian log-likelihoods are summed and normalised
by their observation counts; the two streams are combined with the
``alpha_h, alpha_v`` weights from ``InferenceConfig.stream_weights``
(prototype defaults are 0.5/0.5). The combined log-lik is registered as a
``pm.Potential``; ``w`` is exposed as a deterministic softmax of the
scaled log-lik.

Stream weights are pre-multiplied outside the PyMC graph for clarity. If
they should be embedded deeper, move the multiplication inside the loop
so each per-member ``logL_thick``/``logL_vel`` already carries its weight.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pymc as pm

from fusion.config import InferenceConfig
from fusion.data.prepare import PreparedData

if TYPE_CHECKING:
    import arviz as az


def run_inference(
    prepared: PreparedData,
    cfg: InferenceConfig,
    random_seed: int = 42,
    progressbar: bool = True,
) -> az.InferenceData:
    """Sample the hierarchical model and return the trace.

    Parameters
    ----------
    prepared
        Output of :func:`fusion.data.prepare.prepare`.
    cfg
        Inference settings — uses ``stream_weights``, ``draws``, ``tune``,
        ``chains``, ``target_accept``.
    random_seed
        Forwarded to ``pm.sample`` so the MCMC chain is reproducible.
        The prototype hardcodes 42 — match it for the validation harness.
    progressbar
        Show PyMC's per-chain tqdm progress bar. Defaults to ``True``
        so interactive callers (CLI, the validation harness) see
        progress; tests can pass ``False`` to keep CI output clean.
    """
    model = _build_model(prepared, cfg)
    with model:
        # pm.sample is untyped; with return_inferencedata=True it yields an
        # arviz InferenceData. Pin the type at this boundary.
        trace: az.InferenceData = pm.sample(
            draws=cfg.draws,
            tune=cfg.tune,
            chains=cfg.chains,
            target_accept=cfg.target_accept,
            return_inferencedata=True,
            random_seed=random_seed,
            progressbar=progressbar,
        )
    return trace


def _build_model(prepared: PreparedData, cfg: InferenceConfig) -> pm.Model:
    # PreparedData arrays are already float64 (see fusion.data.prepare).
    y = prepared.y_obs
    sig_obs = prepared.sigma_obs
    speed = prepared.speed
    F = prepared.F
    n_thick = int(prepared.n_dhdt)
    n_vel = int(prepared.n_vel)

    idx_thick = slice(0, n_thick)
    idx_vel = slice(n_thick, n_thick + n_vel)

    alpha_h = cfg.stream_weights.thick
    alpha_v = cfg.stream_weights.vel

    with pm.Model() as model:
        sigma_base_thick = pm.HalfNormal("sigma_base_thick", sigma=0.5)
        sigma_base_vel = pm.HalfNormal("sigma_base_vel", sigma=0.6)
        beta_thick = pm.HalfNormal("beta_thick", sigma=0.1)
        beta_vel = pm.HalfNormal("beta_vel", sigma=0.1)

        sigma_model_thick = sigma_base_thick * pm.math.sqrt(1.0 + beta_thick * speed[idx_thick])
        sigma_model_vel = sigma_base_vel * pm.math.sqrt(1.0 + beta_vel * speed[idx_vel])

        sigma_tot_thick = pm.math.sqrt(sig_obs[idx_thick] ** 2 + sigma_model_thick**2)
        # +5² fixed inflation prevents velocity domination (prototype).
        sigma_tot_vel = pm.math.sqrt(sig_obs[idx_vel] ** 2 + sigma_model_vel**2 + 5.0**2)

        # Vectorised over the M (member) axis. Per-row sum order is
        # preserved vs. the loop version, so this is bit-exact at fixed
        # parameter values (verified by tests/test_inference_vectorised.py).
        R = y - F  # broadcasts to (M, N)
        logL_thick = pm.logp(
            pm.Normal.dist(mu=0.0, sigma=sigma_tot_thick),
            R[:, idx_thick],
        ).sum(axis=1)
        logL_vel = pm.logp(
            pm.Normal.dist(mu=0.0, sigma=sigma_tot_vel),
            R[:, idx_vel],
        ).sum(axis=1)

        N_thick = max(n_thick, 1)
        N_vel = max(n_vel, 1)
        logL_scaled = alpha_h * (logL_thick / N_thick) + alpha_v * (logL_vel / N_vel)

        pm.Potential("joint_loglik", logL_scaled.sum())

        w_unnorm = pm.math.exp(logL_scaled - pm.math.max(logL_scaled))
        w = w_unnorm / pm.math.sum(w_unnorm)

        pm.Deterministic("w", w)
        pm.Deterministic("logL_thick", logL_thick)
        pm.Deterministic("logL_vel", logL_vel)
        pm.Deterministic("logL_scaled", logL_scaled)
    return model
