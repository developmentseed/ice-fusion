import geopandas as gpd  # noqa: F401  (xopr brings this in)

from fusion.config import RegionsConfig
from fusion.data.regions import load_regions


def test_imbie_basins_returns_geodataframe() -> None:
    """When config asks for imbie_basins, we get a GeoDataFrame with name + geometry."""
    cfg = RegionsConfig(source="imbie_basins")
    gdf = load_regions(cfg)
    assert "name" in gdf.columns
    assert "geometry" in gdf.columns
    assert len(gdf) > 0
    assert gdf.crs is not None
