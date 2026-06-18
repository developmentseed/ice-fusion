import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from fusion.config import ObservationsConfig
from fusion.data.obs import load_observations


def test_load_observations_uses_local_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, synthetic_obs_bundle: Path
) -> None:
    """If the cache contains the requested version, no network call is made."""
    cache = tmp_path / "cache"
    versioned = cache / "2025.04"
    versioned.mkdir(parents=True)
    shutil.copytree(synthetic_obs_bundle / "elevation", versioned / "elevation")
    shutil.copytree(synthetic_obs_bundle / "velocity", versioned / "velocity")
    monkeypatch.setenv("FUSION_CACHE", str(cache))

    cfg = ObservationsConfig(source="source-coop", version="2025.04")
    bundle = load_observations(cfg)
    assert set(bundle) == {"elevation", "velocity"}
    assert {"height", "absolute_elevation_rmse"}.issubset(bundle["elevation"].data_vars)
    assert {"VX", "VY", "ERRX", "ERRY"}.issubset(bundle["velocity"].data_vars)
    assert bundle["elevation"].sizes["year"] == 6  # 2015..2020
    assert bundle["velocity"].sizes["year"] == 6  # 2014..2019
    assert bundle["elevation"]["year"].values.tolist() == list(range(2015, 2021))


def test_fetch_bundle_lists_each_stream_via_obspec_glob(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """_fetch_bundle enumerates each stream's prefix via obspec_utils.glob
    (the project's canonical S3 listing primitive) and fetches the keys,
    rather than the raw obstore.list batch loop."""
    patterns: list[str] = []

    def fake_glob(store: object, pattern: str) -> Iterator[str]:
        patterns.append(pattern)
        yield pattern.replace("*", "file.nc")

    class _FakeGet:
        def bytes(self) -> bytes:
            return b""

    monkeypatch.setattr("obspec_utils.glob", fake_glob)
    monkeypatch.setattr("obstore.get", lambda store, key: _FakeGet())
    monkeypatch.setattr("obstore.store.from_url", lambda *a, **k: object())

    from fusion.data.obs import _fetch_bundle

    _fetch_bundle("v0", tmp_path)

    assert patterns == ["v0/elevation/*", "v0/velocity/*"]
    assert (tmp_path / "elevation" / "file.nc").exists()
    assert (tmp_path / "velocity" / "file.nc").exists()
