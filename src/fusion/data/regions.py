"""Region loader.

Wraps :func:`xopr.geometry.get_antarctic_regions` to expose the MEaSUREs
Antarctic basins (NSIDC-0709 v2) as a per-region GeoDataFrame in EPSG:3031
(Antarctic Polar Stereographic — the projection of the 8 km observation
grid) with a normalised lowercase ``name`` column.
"""

import functools

import geopandas as gpd

from fusion.config import RegionsConfig

# The 8 km observation grid is in EPSG:3031, so reproject regions to match.
_ANTARCTIC_CRS = "EPSG:3031"


@functools.lru_cache(maxsize=2)
def _measures_basins() -> gpd.GeoDataFrame:
    from xopr.geometry import get_antarctic_regions

    # merge_regions=False returns one row per basin (the merged form returns
    # a single fused Shapely polygon, which is not what we want here).
    gdf = get_antarctic_regions(merge_regions=False)
    return gdf.to_crs(_ANTARCTIC_CRS)


def load_regions(cfg: RegionsConfig) -> gpd.GeoDataFrame:
    """Return regions as a GeoDataFrame with ``name`` and ``geometry`` columns."""
    if cfg.source == "imbie_basins":
        gdf = _measures_basins().copy()
        if "name" not in gdf.columns:
            if "NAME" in gdf.columns:
                gdf = gdf.rename(columns={"NAME": "name"})
            else:
                gdf = gdf.rename(columns={gdf.columns[0]: "name"})
        return gdf
    raise ValueError(f"Unknown regions source: {cfg.source}")
