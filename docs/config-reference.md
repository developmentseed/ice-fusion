# Config reference

A FUSION run is configured by a single YAML file. [`load_config`][fusion.load_config] reads it, validates it against the [Pydantic][pydantic.BaseModel] models in `fusion.config`, and returns a `Config` object you pass to [`fusion.run`][fusion.run] (or to the individual pipeline steps). A validation error is re-raised as a `ValueError` naming the file, so a malformed config fails fast with the offending field in the message.

Each top-level key configures one part of the pipeline. Start from the complete example, then jump to the section you want to change.

## At a glance

| Key | Pipeline step | Controls |
|-----|---------------|----------|
| [`ensemble`](#ensemble) | `load_data` | Where the PSU-ISM run files live and how to read them. |
| [`observations`](#observations) | `load_data` | Which bundled observation version to score against. |
| [`grid`](#grid) | `load_data` | Target grid and regridding method. |
| [`regions`](#regions) | `prepare` | Region definitions for per-region scoring. |
| [`metric`](#metric) | `prepare` + `sample` | Comparison metric and per-stream uncertainty caps. |
| [`inference`](#inference) | `sample` | PyMC / NUTS sampler settings and likelihood weights. |
| [`projection`](#projection) | `project` | Forward-projection target year and quantity. |

!!! tip "Defaults come from the prototype"
    Most fields have defaults that reproduce the `PSUISM_HBM_V1` prototype bit-exactly. The fields under [`metric`](#metric), [`inference.stream_weights`](#inference), and [`inference.subsample`](#inference) are pinned **science** values. They are surfaced in config so they land in run metadata, not because they are meant to be tuned casually. Changing them breaks the validation baseline. See [Validation](validation.md) and the [Terminology](terminology.md) glossary for *bit-exactness* and *science territory*.

## Complete example

A config with every section filled in with its defaults:

```yaml
--8<-- "tests/data/fixtures/example_config.yaml"
```

Save it as `my_run.yaml`, edit `ensemble.path`, and run it as shown in [Getting started](getting-started.md).

## `ensemble`

Where the ensemble lives on disk and how to read it. Consumed by [`load_data`][fusion.load_data]. `path` is the directory of PSU-ISM run files (one NetCDF per member, each read into an [`xarray.Dataset`][xarray.Dataset]); `adapter` selects the reader.

```yaml
ensemble:
  path: /path/to/psuism/runs/
  adapter: psuism
```

::: fusion.config.EnsembleConfig

## `observations`

Which bundled observations to score against. Consumed by [`load_data`][fusion.load_data]. `version` is an ISO date selecting a published bundle; the first run downloads it from Source.Coop and caches it under `FUSION_CACHE` (later runs are offline). See [File locations](getting-started.md#file-locations).

```yaml
observations:
  source: source-coop
  version: "2026-04-30"
```

::: fusion.config.ObservationsConfig

## `grid`

Target grid the ensemble is regridded onto before scoring. Consumed by [`load_data`][fusion.load_data].

```yaml
grid:
  target: obs_8km
  method: bilinear      # use "conservative" for fluxes / extensive fields
```

!!! note "v1 grid handling"
    v1 enforces that obs and ensemble already share a 761×761 `(y, x)` grid; native-resolution regridding is deferred to v1.1 and raises `NotImplementedError`. `method` is recorded in metadata but does not yet drive a regrid.

::: fusion.config.GridConfig

## `regions`

Region definitions used for per-region scoring and subsampling. Consumed by [`prepare`][fusion.prepare]. `imbie_basins` uses the IMBIE / MEaSUREs NSIDC-0709 v2 basins.

```yaml
regions:
  source: imbie_basins
```

::: fusion.config.RegionsConfig

## `metric`

Which comparison metric to use, plus the per-stream uncertainty caps applied in [`prepare`][fusion.prepare] to drop high-σ pixels. `prepare` finite-differences the gridded fields and flattens them into the 1-D [`numpy.ndarray`][numpy.ndarray] vectors the likelihood sums over. v1 ships only the pixelwise Gaussian likelihood.

```yaml
metric:
  type: pixelwise_gaussian
  thick_unc_threshold: 50.0   # m/yr   (prototype default)
  vel_unc_threshold: 10.0     # m/yr²  (prototype default)
```

!!! warning "Pinned science values"
    `thick_unc_threshold` and `vel_unc_threshold` default to the prototype's hardcoded thresholds. Changing them changes which pixels are masked and breaks Layer 1 bit-exactness against the [validation baseline](validation.md).

::: fusion.config.MetricConfig

## `inference`

The PyMC inference settings. The likelihood weights (`obs_alpha`, `stream_weights`) and the `subsample` are pinned science values; `draws`, `tune`, `chains`, `target_accept`, and `seed` are the standard sampler knobs. Consumed by [`sample`][fusion.sample].

```yaml
inference:
  obs_alpha: 0.5
  stream_weights:
    thick: 0.5
    vel: 0.5
  subsample:
    size: 20000
    seed: 42
  draws: 500
  tune: 1000
  chains: 4
  target_accept: 0.95
  seed: 42
```

**Sampler knobs.** `draws`, `tune`, `chains`, and `seed` map directly onto the same-named arguments of [`pm.sample`][pymc.sample] (`seed` is passed as its `random_seed`). `target_accept` is the NUTS step-size tuning target; raise it toward `0.99` to suppress divergences. The model itself uses half-normal priors ([`pm.HalfNormal`][pymc.HalfNormal]), combines the per-stream log-likelihoods through a [`pm.Potential`][pymc.model.core.Potential], and exposes the member weights as a softmax [`pm.Deterministic`][pymc.model.core.Deterministic]. For what to do when a run does not converge, see [Interpreting results](interpreting-results.md#did-the-run-converge).

!!! warning "Pinned science values"
    `obs_alpha`, `stream_weights`, and `subsample` (`size` / `seed`) default to the prototype's values. They are exposed so they appear in run metadata and so v1.1 can evolve them, not for routine tuning. Changing them breaks bit-exactness. Note `subsample.seed` (the pixel draw) is independent of `inference.seed` (the MCMC seed); both default to `42`.

::: fusion.config.InferenceConfig

::: fusion.config.StreamWeights

::: fusion.config.SubsampleConfig

## `projection`

Which forward-projection target to compute when applying the posterior weights. Consumed by [`project`][fusion.project].

```yaml
projection:
  target_year: 2100
  quantity: grounded_ice_volume
```

!!! warning "Projection magnitudes are provisional"
    The member-to-SLE reduction in v1 is a placeholder (a mean of ice thickness, not volume above flotation). Treat the projection's shape as illustrative, not its absolute magnitude. The weighting itself is validated and unaffected. See [Interpreting results](interpreting-results.md#the-projection).

::: fusion.config.ProjectionConfig

## Loading and validating

[`load_config`][fusion.load_config] wraps Pydantic's [`model_validate`][pydantic.BaseModel.model_validate], so any constraint violation (unknown grid target, wrong type, missing required key) surfaces as a `ValueError` that names the file.

::: fusion.config.load_config
