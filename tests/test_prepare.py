import shutil
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from fusion.config import EnsembleConfig, InferenceConfig, ObservationsConfig, SubsampleConfig
from fusion.data.ensemble import load_ensemble
from fusion.data.obs import load_observations
from fusion.data.prepare import PreparedData, prepare


def _load_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, synthetic_obs_bundle: Path
) -> dict[str, xr.Dataset]:
    cache = tmp_path / "cache"
    versioned = cache / "2025.04"
    versioned.mkdir(parents=True)
    shutil.copytree(synthetic_obs_bundle / "elevation", versioned / "elevation")
    shutil.copytree(synthetic_obs_bundle / "velocity", versioned / "velocity")
    monkeypatch.setenv("FUSION_CACHE", str(cache))
    return load_observations(ObservationsConfig(source="source-coop", version="2025.04"))


def test_prepare_returns_aligned_arrays(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    synthetic_obs_bundle: Path,
    synthetic_psuism_dir: Path,
) -> None:
    obs = _load_bundle(monkeypatch, tmp_path, synthetic_obs_bundle)
    ens = load_ensemble(EnsembleConfig(path=synthetic_psuism_dir, adapter="psuism"))
    cfg = InferenceConfig(subsample=SubsampleConfig(size=10_000, seed=42))

    prepared = prepare(obs, ens, cfg)

    assert isinstance(prepared, PreparedData)
    n = prepared.y_obs.size
    assert prepared.sigma_obs.size == n
    assert prepared.speed.size == n
    assert prepared.F.shape == (ens.sizes["member"], n)
    assert prepared.n_dhdt + prepared.n_vel == n
    assert np.isfinite(prepared.y_obs).all()
    assert np.isfinite(prepared.speed).all()
    assert prepared.member_ids == sorted(ens["member"].values.tolist())


def test_prepare_subsample_seed_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    synthetic_obs_bundle: Path,
    synthetic_psuism_dir: Path,
) -> None:
    obs = _load_bundle(monkeypatch, tmp_path, synthetic_obs_bundle)
    ens = load_ensemble(EnsembleConfig(path=synthetic_psuism_dir, adapter="psuism"))
    cfg_a = InferenceConfig(subsample=SubsampleConfig(size=200, seed=42))
    cfg_b = InferenceConfig(subsample=SubsampleConfig(size=200, seed=99))

    p1 = prepare(obs, ens, cfg_a)
    p2 = prepare(obs, ens, cfg_a)
    p3 = prepare(obs, ens, cfg_b)

    np.testing.assert_array_equal(p1.y_obs, p2.y_obs)
    assert not np.array_equal(p1.y_obs, p3.y_obs)


def test_prepare_thresholds_drop_high_uncertainty_pixels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    synthetic_obs_bundle: Path,
    synthetic_psuism_dir: Path,
) -> None:
    """Synthetic uncertainty (5.0 elev, 2.0 vel) is below the thresholds —
    every pixel should survive (no NaNs in the source data)."""
    obs = _load_bundle(monkeypatch, tmp_path, synthetic_obs_bundle)
    ens = load_ensemble(EnsembleConfig(path=synthetic_psuism_dir, adapter="psuism"))
    cfg = InferenceConfig(subsample=SubsampleConfig(size=1_000_000, seed=42))

    prepared = prepare(obs, ens, cfg)

    # Synthetic ensemble has 7 cleaned time steps → 6 intervals.
    # Tol=1.5 yrs means every model year snaps to a valid obs year:
    #   thickness: 6 intervals × 100 px = 600
    #   velocity: 6 intervals × 200 (vx then vy) = 1200
    assert prepared.n_dhdt == 600
    assert prepared.n_vel == 1200
