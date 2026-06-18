"""Characterisation test for pipeline.plug_in_weights vectorisation."""

import numpy as np
import xarray as xr
from arviz import InferenceData

from fusion.data.prepare import PreparedData
from fusion.pipeline import plug_in_weights


def _toy_prepared() -> PreparedData:
    rng = np.random.default_rng(0)
    n_thick, n_vel, M = 80, 120, 3
    n = n_thick + n_vel
    return PreparedData(
        y_obs=rng.normal(0, 1, n).astype(float),
        sigma_obs=np.full(n, 0.5),
        F=rng.normal(0, 1, (M, n)).astype(float),
        speed=np.concatenate([np.zeros(n_thick), rng.uniform(0, 100, n_vel)]),
        member_ids=[f"run{i + 1}" for i in range(M)],
        n_dhdt=n_thick,
        n_vel=n_vel,
    )


def _fake_trace() -> InferenceData:
    """Minimal arviz InferenceData with the four scalars plug_in_weights needs."""
    posterior = xr.Dataset(
        {
            "sigma_base_thick": (("chain", "draw"), [[0.4]]),
            "sigma_base_vel": (("chain", "draw"), [[0.5]]),
            "beta_thick": (("chain", "draw"), [[0.05]]),
            "beta_vel": (("chain", "draw"), [[0.07]]),
        }
    )
    return InferenceData(posterior=posterior)


# Snapshot literals captured from the loop version. SNAPSHOT_LL is the
# N-scaled loglik (raw sum / y_obs.size, here n=200) — the value
# plug_in_weights returns and that the prototype writes to its
# model_weights_table.csv ``log_likelihood`` column.
SNAPSHOT_W = np.array([0.26953230694470987, 0.3475843300741724, 0.3828833629811178])
SNAPSHOT_LL = np.array([-2.457122850892556, -2.202803799366642, -2.106080703899742])


def test_plug_in_weights_snapshot():
    """Locks (weights, loglik) to literals captured from the loop version."""
    prepared = _toy_prepared()
    trace = _fake_trace()
    w, ll = plug_in_weights(prepared, trace)
    np.testing.assert_array_equal(w, SNAPSHOT_W)
    np.testing.assert_array_equal(ll, SNAPSHOT_LL)
