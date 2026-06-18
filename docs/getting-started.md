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

## File locations

`ice-fusion` reads and writes files in three places:

**Ensemble input.** `ensemble.path` in your config points at the directory of PSU-ISM run files (one NetCDF per member). This is the only input path you set yourself.

**Observation cache.** The bundled observations download to `~/.cache/fusion/<version>/` on first use. Every later run reads from there. Override the location with the `FUSION_CACHE` environment variable:

```bash
export FUSION_CACHE=/scratch/$USER/fusion-cache
```

Set this when `~/.cache` is small, non-persistent, or shared read-only. This is common on HPC login nodes and in CI. The `<version>` subdirectory comes from `observations.version` in your config, so multiple versions coexist without clobbering each other.

**Outputs.** [`save_weights`][fusion.save_weights], [`plot_projection`][fusion.plot_projection], and [`save_metadata`][fusion.save_metadata] write to whatever path you pass them (relative paths resolve against the current working directory).
