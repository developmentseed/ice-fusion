# FUSION

A Python package for scoring ice-sheet model ensembles against modern satellite observations, and weighting future projections by historical skill.

`FUSION` takes an ensemble of ice-sheet model runs, scores each member against modern satellite observations over the historical period, and produces a weighted projection of sea-level contribution to 2100.

```bash
pip install ice-fusion
```

```python
import fusion
cfg = fusion.load_config("my_run.yaml")
result = fusion.run(cfg)
fusion.save_weights(result, "weights.csv")
```

See [Getting started](getting-started.md) for the full walkthrough, or jump to the [API reference](api/fusion.md).
