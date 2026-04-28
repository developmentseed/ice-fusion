from pathlib import Path

import pytest

from fusion.config import Config, load_config

FIXTURE = Path(__file__).parent / "data" / "fixtures" / "example_config.yaml"

def test_load_example_config():
    cfg = load_config(FIXTURE)
    assert isinstance(cfg, Config)
    assert cfg.ensemble.adapter == "psuism"
    assert cfg.grid.target == "obs_8km"
    assert cfg.metric.type == "pixelwise_gaussian"
    assert cfg.inference.stream_weights.thick == 1.5
    assert cfg.inference.stream_weights.vel == 1.5
    assert cfg.inference.subsample.size == 20_000
    assert cfg.inference.subsample.seed == 42
    assert cfg.projection.target_year == 2100

def test_unknown_metric_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(
        "ensemble: {path: /tmp, adapter: psuism}\n"
        "observations: {source: source-coop, version: '2025.04'}\n"
        "grid: {target: obs_8km, method: conservative}\n"
        "regions: {source: imbie_basins}\n"
        "metric: {type: not_a_real_metric}\n"
        "inference: {stream_weights: {thick: 1.5, vel: 1.5}}\n"
        "projection: {target_year: 2100, quantity: grounded_ice_volume}\n"
    )
    with pytest.raises(ValueError, match="metric"):
        load_config(p)
