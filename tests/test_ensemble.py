import xarray as xr

from fusion.config import EnsembleConfig
from fusion.data.ensemble import load_ensemble


def test_psuism_adapter_drops_spurious_time_and_decodes_years(synthetic_psuism_dir):
    cfg = EnsembleConfig(path=synthetic_psuism_dir, adapter="psuism")
    ds = load_ensemble(cfg)
    assert isinstance(ds, xr.Dataset)
    assert ds.sizes["member"] == 2
    # spurious 1e30 row must be gone — fixture has 7 real years + 1 bogus
    assert ds.sizes["time"] == 7
    assert {"h", "ua", "va"}.issubset(ds.data_vars)
    # decimal years span 2014..2020
    assert ds["time"].min().item() < 2015
    assert ds["time"].max().item() > 2019
    # member ids extracted from runNN token, not the raw file stem
    assert set(ds["member"].values.tolist()) == {"run01", "run07"}
