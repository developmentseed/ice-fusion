"""Characterisation test for inference/model.py vectorisation.

Pins the joint logp of `_build_model` at a deterministic parameter point
so the per-member loop → broadcast `pm.logp` refactor can be verified
bit-exactly. The joint logp is the sum over members of per-row-summed
log-pdfs, so it is sensitive to changes in either summation order.
"""

import numpy as np

from fusion.config import InferenceConfig, StreamWeights
from fusion.data.prepare import PreparedData
from fusion.inference.model import _build_model

# Snapshot literal captured from the loop version. Update only when the
# model definition is intentionally changed.
SNAPSHOT_LOGP = -11.727317224364928


def _toy(M: int = 3, n_thick: int = 80, n_vel: int = 120, seed: int = 0) -> PreparedData:
    rng = np.random.default_rng(seed)
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


def _fixed_point(M: int) -> dict[str, float]:
    return {
        "sigma_base_thick_log__": np.log(0.4),
        "sigma_base_vel_log__": np.log(0.5),
        "beta_thick_log__": np.log(0.05),
        "beta_vel_log__": np.log(0.07),
    }


def test_build_model_logp_at_fixed_point() -> None:
    """Snapshot the joint logp at a deterministic parameter point."""
    prepared = _toy()
    cfg = InferenceConfig(stream_weights=StreamWeights(thick=0.5, vel=0.5))
    model = _build_model(prepared, cfg)
    point = _fixed_point(prepared.F.shape[0])
    val = float(model.compile_logp()(point))
    np.testing.assert_array_equal(val, SNAPSHOT_LOGP)
