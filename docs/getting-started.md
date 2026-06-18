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

## Step-by-step

For users who want to inspect intermediate state or rerun later steps with different parameters:

```python
import fusion

cfg  = fusion.load_config("my_run.yaml")
data = fusion.load_data(cfg)            # fetch obs + load ensemble (8 km grid)
prepared = fusion.prepare(cfg, data)    # rate-of-change + flatten + mask
trace = fusion.sample(cfg, prepared)    # PyMC inference
proj = fusion.project(cfg, trace, data)
```

Each function takes the same `cfg` object and returns a value you can save and reload.

## Outputs

- **Weights CSV** ([`save_weights`][fusion.save_weights]): one row per ensemble member. Holds both the Bayesian posterior weight (with uncertainty) and the plug-in point estimate.
- **SLE projection plot** ([`plot_projection`][fusion.plot_projection]): weighted distribution of sea-level contribution in 2100.
- **`run_metadata.json`** ([`save_metadata`][fusion.save_metadata]): version, resolved config, obs version, seeds, file hashes.

These write to whatever path you pass them (relative paths resolve against the current working directory).

## File locations

`ice-fusion` reads files from two places (outputs go wherever you pass them, as above):

**Ensemble input.** `ensemble.path` in your config points at the directory of PSU-ISM run files (one NetCDF per member). This is the only input path you set yourself.

**Observation cache.** The bundled observations download to `~/.cache/fusion/<version>/` on first use. Every later run reads from there. Override the location with the `FUSION_CACHE` environment variable:

```bash
export FUSION_CACHE=/scratch/$USER/fusion-cache
```

Set this when `~/.cache` is small, non-persistent, or shared read-only. This is common on HPC login nodes and in CI. The `<version>` subdirectory comes from `observations.version` in your config, so multiple versions coexist without clobbering each other.
