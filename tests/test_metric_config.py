"""Tests for MetricConfig.thick_unc_threshold / vel_unc_threshold."""

from fusion.config import MetricConfig


def test_metric_config_default_thresholds_match_prototype() -> None:
    """Defaults must equal the prototype's hardcoded values, or the
    validation harness will diverge bit-exactly."""
    cfg = MetricConfig(type="pixelwise_gaussian")
    assert cfg.thick_unc_threshold == 50.0
    assert cfg.vel_unc_threshold == 10.0


def test_metric_config_thresholds_user_overridable() -> None:
    cfg = MetricConfig(type="pixelwise_gaussian", thick_unc_threshold=25.0, vel_unc_threshold=5.0)
    assert cfg.thick_unc_threshold == 25.0
    assert cfg.vel_unc_threshold == 5.0
