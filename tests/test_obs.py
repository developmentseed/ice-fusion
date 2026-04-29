import shutil

from fusion.config import ObservationsConfig
from fusion.data.obs import load_observations


def test_load_observations_uses_local_cache(monkeypatch, tmp_path, synthetic_obs_bundle):
    """If the cache contains the requested version, no network call is made."""
    cache = tmp_path / "cache"
    versioned = cache / "2025.04"
    versioned.mkdir(parents=True)
    shutil.copytree(synthetic_obs_bundle / "elevation", versioned / "elevation")
    shutil.copytree(synthetic_obs_bundle / "velocity", versioned / "velocity")
    monkeypatch.setattr("fusion.data.obs.CACHE_DIR", cache)

    cfg = ObservationsConfig(source="source-coop", version="2025.04")
    bundle = load_observations(cfg)
    assert set(bundle) == {"elevation", "velocity"}
    assert {"height", "absolute_elevation_rmse"}.issubset(bundle["elevation"].data_vars)
    assert {"VX", "VY", "ERRX", "ERRY"}.issubset(bundle["velocity"].data_vars)
    assert bundle["elevation"].sizes["year"] == 6  # 2015..2020
    assert bundle["velocity"].sizes["year"] == 6  # 2014..2019
    assert bundle["elevation"]["year"].values.tolist() == list(range(2015, 2021))
