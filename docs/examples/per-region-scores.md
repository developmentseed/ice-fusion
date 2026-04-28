# Per-region scores

To inspect how each member performs in each MEaSUREs Antarctic basin:

```python
import fusion

cfg = fusion.load_config("my_run.yaml")
data = fusion.load_data(cfg)
scores = fusion.score(cfg, data)   # shape: (n_members, n_regions)

import xarray as xr
xr.DataArray(
    scores,
    dims=("member", "region"),
    coords={"member": data["ensemble"].member, "region": data["regions"].name},
).to_netcdf("per_region_scores.nc")
```

Regions come from `xopr.geometry.get_antarctic_regions()` (NSIDC-0709 v2). See the [config reference](../config-reference.md) for swapping the source.
