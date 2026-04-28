# Getting started

## Install

```bash
pip install ice-fusion
```

## Scaffold a config

```bash
fusion init my_run.yaml
```

This drops a config with sensible defaults:

```yaml
--8<-- "tests/data/fixtures/example_config.yaml"
```

Edit the `ensemble.path` to point at your PSU-ISM run directory, then:

```python
import fusion
cfg = fusion.load_config("my_run.yaml")
result = fusion.run(cfg)
fusion.save_weights(result, "weights.csv")
fusion.plot_projection(result, "sle_2100.png")
fusion.save_metadata(result, "run_metadata.json")
```

The first call downloads the bundled observations from Source Co-op and caches them locally; subsequent runs are offline.
