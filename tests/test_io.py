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
