import numpy as np

from fusion.config import InferenceConfig, StreamWeights
from fusion.data.prepare import PreparedData
from fusion.inference.model import run_inference


def _toy_prepared(M: int = 3, n_thick: int = 80, n_vel: int = 120, seed: int = 0) -> PreparedData:
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


def test_inference_runs_and_exposes_w_and_priors():
    prepared = _toy_prepared()
    cfg = InferenceConfig(
        stream_weights=StreamWeights(thick=0.5, vel=0.5),
        draws=30,
        tune=30,
        chains=2,
        target_accept=0.9,
    )
    trace = run_inference(prepared, cfg, random_seed=42)
    assert "w" in trace.posterior
    assert trace.posterior["w"].shape[-1] == prepared.F.shape[0]
    # Prior parameters surface as posterior variables.
    for name in ("sigma_base_thick", "sigma_base_vel", "beta_thick", "beta_vel"):
        assert name in trace.posterior
    # Per-stream loglikelihoods are exposed for diagnostics.
    assert "logL_thick" in trace.posterior
    assert "logL_vel" in trace.posterior


def test_inference_random_seed_is_deterministic():
    """Same seed → identical w posterior; different seed → different."""
    prepared = _toy_prepared()
    cfg = InferenceConfig(
        stream_weights=StreamWeights(thick=0.5, vel=0.5),
        draws=20,
        tune=20,
        chains=2,
        target_accept=0.9,
    )
    a = run_inference(prepared, cfg, random_seed=42).posterior["w"].values
    b = run_inference(prepared, cfg, random_seed=42).posterior["w"].values
    c = run_inference(prepared, cfg, random_seed=99).posterior["w"].values
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
