import shutil

import pytest

from fusion import load_config, run


@pytest.mark.slow
def test_pipeline_end_to_end(synthetic_psuism_dir, synthetic_obs_bundle, tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    versioned = cache / "2025.04"
    versioned.mkdir(parents=True)
    shutil.copytree(synthetic_obs_bundle / "elevation", versioned / "elevation")
    shutil.copytree(synthetic_obs_bundle / "velocity", versioned / "velocity")
    monkeypatch.setenv("FUSION_CACHE", str(cache))

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        f"""
ensemble: {{path: {synthetic_psuism_dir}, adapter: psuism}}
observations: {{source: source-coop, version: '2025.04'}}
grid: {{target: obs_8km, method: bilinear}}
regions: {{source: imbie_basins}}
metric: {{type: pixelwise_gaussian}}
inference:
  obs_alpha: 0.5
  stream_weights: {{thick: 0.5, vel: 0.5}}
  subsample: {{size: 5000, seed: 42}}
  draws: 30
  tune: 30
  chains: 2
projection: {{target_year: 2100, quantity: grounded_ice_volume}}
"""
    )
    cfg = load_config(cfg_path)
    result = run(cfg)
    assert result.prepared is not None
    assert result.weights is not None
    assert len(result.weights) == 2  # synthetic fixture has 2 members
    assert {"posterior_mean", "posterior_sd", "point_estimate"}.issubset(result.weights.columns)
    assert result.projection is not None
    assert result.projection.sizes["sample"] == 30 * 2  # draws × chains
