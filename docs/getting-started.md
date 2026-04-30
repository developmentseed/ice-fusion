# Getting started

## Install

```bash
pip install ice-fusion
```

## Write a config

A config with sensible defaults:

```yaml
--8<-- "tests/data/fixtures/example_config.yaml"
```

Save that as `my_run.yaml`, edit `ensemble.path` to point at your PSU-ISM run directory, then:

```python
import fusion
cfg = fusion.load_config("my_run.yaml")
result = fusion.run(cfg)
fusion.save_weights(result, "weights.csv")
fusion.plot_projection(result, "sle_2100.png")
fusion.save_metadata(result, "run_metadata.json")
```

The first call downloads the bundled observations from Source and caches them locally; subsequent runs are offline.
