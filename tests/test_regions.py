import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from fusion.config import RegionsConfig
from fusion.data import regions as regions_mod
from fusion.data.regions import load_regions


def _fake_basins() -> gpd.GeoDataFrame:
    """Minimal stand-in for ``xopr.geometry.get_antarctic_regions`` output.

    Two basins in EPSG:4326 with an upper-case ``NAME`` column, so the test
    exercises both the column normalisation and the reprojection in
    ``load_regions`` without a live network fetch.
    """
    return gpd.GeoDataFrame(
        {"NAME": ["Basin-A", "Basin-B"]},
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1)]),
            Polygon([(1, 1), (2, 1), (2, 2)]),
        ],
        crs="EPSG:4326",
    )


def test_imbie_basins_returns_geodataframe(monkeypatch: pytest.MonkeyPatch) -> None:
    """imbie_basins yields a GeoDataFrame with name + geometry in EPSG:3031."""
    import xopr.geometry

    monkeypatch.setattr(xopr.geometry, "get_antarctic_regions", lambda **kw: _fake_basins())
    # _measures_basins is lru_cache'd; clear it so the mock is used and does
    # not leak a cached value into (or out of) this test.
    regions_mod._measures_basins.cache_clear()

    cfg = RegionsConfig(source="imbie_basins")
    gdf = load_regions(cfg)

    assert "name" in gdf.columns  # NAME normalised to name
    assert "geometry" in gdf.columns
    assert len(gdf) == 2
    assert gdf.crs is not None
    assert gdf.crs.to_epsg() == 3031  # reprojected to the 8 km obs grid CRS

    regions_mod._measures_basins.cache_clear()
