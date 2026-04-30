import json

import pandas as pd

from fusion import save_metadata, save_weights
from fusion.result import Result


def test_save_weights_roundtrip(tmp_path):
    df = pd.DataFrame({"member_id": ["a", "b"], "posterior_mean": [0.5, 0.5]})
    res = Result(config=None, weights=df)
    out = tmp_path / "w.csv"
    save_weights(res, out)
    assert pd.read_csv(out).equals(df)


def test_save_metadata_includes_version_and_seed(tmp_path):
    res = Result(config=None, metadata={"obs_version": "2025.04", "seed": 42})
    out = tmp_path / "meta.json"
    save_metadata(res, out)
    payload = json.loads(out.read_text())
    assert payload["fusion_version"]
    assert payload["environment"]["python"]
    assert payload["obs_version"] == "2025.04"
    assert payload["seed"] == 42


def test_save_metadata_dumps_resolved_config(tmp_path):
    """Validation harness reads obs version + seed + stream weights off config."""
    from pathlib import Path as _P

    from fusion.config import load_config

    fixture = _P(__file__).parent / "data" / "fixtures" / "example_config.yaml"
    cfg = load_config(fixture)
    out = tmp_path / "meta.json"
    save_metadata(Result(config=cfg), out)
    payload = json.loads(out.read_text())
    assert payload["config"]["observations"]["version"] == "2025.04"
    assert payload["config"]["inference"]["subsample"]["seed"] == 42
    assert payload["config"]["inference"]["stream_weights"]["thick"] == 0.5
    assert payload["config"]["inference"]["stream_weights"]["vel"] == 0.5
