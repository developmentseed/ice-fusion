# Contributing

Welcome, and thank you for picking this up. This page is the handoff guide for taking over `ice-fusion`.

The package itself is on PyPI as `ice-fusion`; the import name is `fusion`. Source lives at [developmentseed/ice-fusion](https://github.com/developmentseed/ice-fusion) on GitHub, but will be transferred.

Publishing a new observation bundle is a separate maintainer task with its own guide: see [Data upload to Source.Coop](data-upload.md).

## What you're inheriting

`ice-fusion` ports Sara Peters' [`PSUISM_HBM_V1`](https://github.com/sc-peters/PSUISM_HBM_V1) prototype (`full_model.py`) into a packaged library. Some parts are "Science territory": the metric definition, the priors, and the per-stream weights. These are deliberately marked as such in the source:

- `src/fusion/inference/model.py`: the hierarchical Bayesian model, ported from `build_model_proposal` in the prototype.
- `src/fusion/data/prepare.py`: the rate-of-change preparation that builds the inputs the PyMC model consumes.

The validation harness in `validation/` is what keeps the port honest.

## Dev setup

The project uses [`uv`](https://docs.astral.sh/uv/) for dependency management and Python 3.12+.

```bash
git clone https://github.com/developmentseed/ice-fusion.git
cd ice-fusion
uv sync --all-extras
```

Run the test suite:

```bash
uv run pytest
```

Slow tests (PyMC end-to-end) are gated behind `-m slow`:

```bash
uv run pytest -m slow
```

## Repository tour

| Path | What lives there |
|------|------------------|
| `src/fusion/` | Library code |
| `src/fusion/inference/model.py` | PyMC model definition (Science territory) |
| `src/fusion/data/prepare.py` | Rate-of-change preparation (Science territory) |
| `src/fusion/data/obs.py` | Observation fetcher with on-disk cache |
| `src/fusion/data/ensemble.py` | PSU-ISM ensemble adapter |
| `src/fusion/pipeline.py` | Top-level `load_data → prepare → sample → project` |
| `validation/` | Manual side-by-side harness (not CI) |
| `validation/baseline/full_model.py` | Pinned fork of the prototype |
| `validation/compare.py` | Driver that runs both stacks and diffs them |
| `validation/reports/` | Signed-off comparison reports, one per release |
| `tests/` | Unit + integration tests |
| `docs/` | mkdocs source for the user-facing docs site |

## Assumptions and how to loosen them

This section catalogs the assumptions v1 bakes in, what happens if an input violates them, and the lever to relax each one. Many are deliberate v1 simplifications with a planned v1.1 escape hatch.

**Read this first if you intend to change anything in "Science territory"** (`data/prepare.py`, `inference/model.py`) or the masking / threshold / subsample defaults. Those values are pinned for **bit-exactness** against the validation baseline. Changing them is legitimate for v1.1, but you must re-run `uv run python -m validation.compare` and obtain a fresh sign-off. If the change alters the metric *definition*, refresh the baseline too. They were surfaced into config so the values land in run metadata, **not** because they are meant to be changed casually.

### Spatial grid

| Assumption | Where | If violated | How to loosen |
|---|---|---|---|
| Obs and ensemble share an identical `(y, x)` shape (761×761 for the reference inputs). | `pipeline._check_grid_compatible` | Mismatched shapes raise `NotImplementedError`. | Implement the deferred v1.1 regridding (below). |
| The comparison is purely **positional**. Coordinate *values* are never compared, only array shape. | `pipeline._check_grid_compatible` | Two inputs that share a shape but cover different extents / orientations / projections silently compare unrelated pixels and yield meaningless weights. There is **no error**. | Add coord-value validation and/or regridding so pixels are physically aligned before [`prepare`][fusion.prepare]. |

Native-resolution regridding is the intended v1.1 escape hatch: regrid obs and ensemble onto a common grid (coord-value-aware, via the `grid.method` `bilinear`/`conservative` knob that already exists on `GridConfig` but is currently unused) inside [`load_data`][fusion.load_data], then drop the shape-only guard.

### Observation bundle

| Assumption | Where | If violated | How to loosen |
|---|---|---|---|
| Stream subdirectories are named exactly `elevation/` and `velocity/`. | `obs.load_observations` (`versioned / "elevation"`, `/ "velocity"`) | `FileNotFoundError` (no NetCDFs found). | Parametrize the stream→subdir mapping in the loader (or surface it on `ObservationsConfig`). |
| Each file's year is the **first** four-digit run in its filename. | `obs._stack_yearly` (regex `(\d{4})`) | No 4-digit run → `ValueError`; a non-year 4-digit token (e.g. a resolution like `1000`) matches first → silently mislabeled year. | Tighten the regex, or pass an explicit filename→year map. |
| Variables are named exactly `height`, `absolute_elevation_rmse` (elevation) and `VX`, `VY`, `ERRX`, `ERRY` (velocity). | `obs._stack_yearly`, `data/prepare.py` | `KeyError`. | Add a rename map in the loader, or make the variable names configurable. |
| An existing `<cache>/<version>/` directory is a complete, trustworthy bundle. | `obs.load_observations` (download skipped if it exists) | A partial or corrupt cached version is used as-is. There is no integrity check. | Add a manifest/checksum check before trusting the cache; delete the dir to force a re-fetch. |

### Ensemble

| Assumption | Where | If violated | How to loosen |
|---|---|---|---|
| Only the `psuism` adapter exists. | `ensemble.load_ensemble`, `EnsembleConfig.adapter` (`Literal`) | Pydantic rejects any other value; `load_ensemble` raises on an unknown adapter. | Write a new adapter function and add its name to the `adapter` `Literal`. |
| Member id is the `runNN` token in the filename, else the file stem. | `ensemble._member_id` | Files lacking `runNN` fall back to the stem; two files with the same id produce **duplicate member labels**, breaking concat/selection. | Change the regex, or supply explicit member ids. |
| Ensemble variables are named exactly `h`, `ua`, `va`. | `ensemble.py`, `data/prepare.py` | `KeyError` in [`prepare`][fusion.prepare]. | Rename in the adapter's `_preprocess`. |

### Time axis and alignment

| Assumption | Where | If violated | How to loosen |
|---|---|---|---|
| All members share one time axis; the reference axis is taken from **member 0**. | `prepare.prepare` (`t0`), adapter `join="outer"` | Members with differing axes are outer-joined (introducing NaNs) and then evaluated against member 0's intervals → misaligned rates / dropped pixels, **no hard error**. | Compute rates per member on each member's own axis before stacking. |
| Ensemble has ≥ 2 time steps with no zero/duplicate `dt`. | `prepare.prepare` | `ValueError` (intentional guard). | n/a. This is a correctness guard, not a limitation. |
| Time units are `seconds since …` or already decimal years; anything else is **assumed** to already be decimal years. | `time_utils.model_decimal_years_from_ds` | An unrecognized calendar/unit is silently treated as decimal years → wrong times. | Extend `model_decimal_years_from_ds` to handle the new units. |
| A model year matches an obs year only within `tol = 1.5` years of an interval endpoint, else that interval is dropped. | `time_utils.snap_model_year_to_obs_year` | Obs/model temporal offsets > 1.5 yr silently drop intervals (fewer obs, possibly none). | Pass a different `tol`. |

### Metric and preparation: Science territory (bit-exact)

| Assumption | Where | If violated | How to loosen |
|---|---|---|---|
| Per-stream uncertainty caps: drop pixels with σ ≥ `50.0` (thickness) / `10.0` (velocity). | `MetricConfig` defaults, `prepare._flatten_and_mask_combined` | Breaks Layer-1 bit-exactness; lower drops data, higher admits noisy pixels. | Set `metric.thick_unc_threshold` / `metric.vel_unc_threshold` in config. |
| Thickness-uncertainty NaN fill: all-NaN → constant `20.0` m/yr; partial-NaN → median of finite values. | `prepare._fill_thickness_unc` (the `20.0` is hardcoded) | The reference data is 100% NaN so the constant-20 branch always fires; real σ would take the median branch instead. | Parametrize the constant/strategy (currently source-only). |
| Final subsample is `size = 20000`, `seed = 42`, drawn as a **single global** sample. | `SubsampleConfig` defaults, `prepare.prepare` | Different size/seed → different pixel set → breaks bit-exactness. | Set `inference.subsample.size` / `.seed`. |
| dtype discipline: storage float32, promoted to float64 at the boundary; observed speed computed in float32. | `data/prepare.py` | Changing dtype shifts Layer-1 results (~1e-4 m/yr on `speed` alone). | Only alongside a refreshed baseline. |

### Bayesian model: Science territory (bit-exact)

| Assumption | Where | If violated | How to loosen |
|---|---|---|---|
| Priors are hardcoded: `HalfNormal` σ = 0.5 / 0.6 (`sigma_base_thick`/`_vel`), 0.1 / 0.1 (`beta_*`). | `model._build_model` (not in config) | Editing changes inference and breaks Layer 3. | Surface as config fields and thread into `_build_model`; re-validate. |
| Velocity stream gets a fixed `+5²` variance inflation. | `model._build_model` | Hardcoded; changing alters the velocity/thickness balance. | Parametrize alongside the priors. |
| Likelihood is zero-mean Gaussian on residuals, per-stream loglik normalized by N, combined via `stream_weights` (`0.5 / 0.5`), weights = softmax. Only `pixelwise_gaussian` exists. | `model._build_model`, `MetricConfig.type` (`Literal`) | This *is* the metric; only `stream_weights` is config-exposed. | Adjust `inference.stream_weights`; for a different form, add a new metric `type` and branch (a substantial science change). |
| Per-row summation order is preserved to stay bit-exact with the prototype's loops. | `model._build_model`, `pipeline.plug_in_weights` | Reordering breaks Layer-2 bit-exactness. | Don't reorder unless refreshing the baseline (see `tests/test_inference_vectorised.py`, `tests/test_plug_in_weights.py`). |

### Projection

| Assumption | Where | If violated | How to loosen |
|---|---|---|---|
| `_sle_per_member` is a **placeholder**: mean of `h` over `(x, y, time)`. It ignores `target_year` and is **not release-quality**. | `pipeline._sle_per_member` | Projection numbers are physically meaningless. This is a known open item, not a bug to paper over. | Implement the science-owned volume-above-flotation → SLE reduction. `compute_projection` is already agnostic (aligns by member label) and needs no change. |
| Only `grounded_ice_volume` is a valid projection quantity. | `ProjectionConfig.quantity` (`Literal`) | Pydantic rejects other values. | Add the quantity to the `Literal` and implement its reduction. |

### Config knobs that are accepted but not yet wired

These validate and land in run metadata, but **do not affect results in v1**. Don't assume setting them changes anything:

- **`regions` (`imbie_basins`):** `load_regions` (`data/regions.py`) is never called, and [`prepare`][fusion.prepare] does a single global subsample, *not* a per-region one (despite `SubsampleConfig`'s "within each region" docstring). To wire it up: call `load_regions` in [`load_data`][fusion.load_data], mask pixels per basin, and subsample per region. Note `load_regions` fetches live from `xopr` (network + `xopr` required at call time).
- **`obs_alpha`:** present on `InferenceConfig` but not consumed by the model (only `stream_weights` feed the likelihood). Reserved for a v1.1 Dirichlet weighting scheme.
- **`grid.method`:** accepted but unused until regridding lands (see [Spatial grid](#spatial-grid)).

## Validation against the prototype

This is the gate that releases v1. See [Validation](validation.md) for the full process. In short:

1. Drop reference inputs into `validation/data/` (gitignored; see `validation/baseline/README.md` for the layout).
2. Run the harness:
   ```bash
   uv run python -m validation.compare
   ```
3. The harness writes `validation/reports/<YYYY-MM-DD>.md`. The report contains a sign-off block with two checkboxes (Max + Sara). Only your sign-off releases a version, not a green CI run.

The comparison runs at three layers:

- **Layer 1**, prepared arrays (`y_obs`, `sigma_obs`, `F`, `speed`, `n_dhdt`, `n_vel`): bit-exact, `np.array_equal`.
- **Layer 2**, per-member plug-in log-likelihood at a canonical posterior mean: bit-exact.
- **Layer 3**, posterior summaries (`sigma_base_*`, `beta_*`, `w`): `rtol=1e-3`.

Re-run validation any time the metric or PyMC model changes, and any time the upstream prototype is refreshed into `validation/baseline/`.

## Refreshing the baseline from upstream

You own the canonical prototype upstream at `sc-peters/PSUISM_HBM_V1`. When the upstream version changes meaningfully, refresh `validation/baseline/full_model.py` so the harness compares against current science:

1. Re-download `full_model.py` from the upstream repo.
2. Re-apply the env-var-override patches listed in `validation/baseline/README.md` (paths, subsample seed, subsample size). They're marked inline with `# PATCH (validation/baseline): ...` comments.
3. Run `uv run python -m validation.compare`.
4. Commit the refreshed baseline and the new validation report together so the diff is reviewable.

If the refreshed baseline breaks Layer 1 or Layer 2, the port needs an update. That's the signal to change `src/fusion/`.

## Releasing

Versioning is automatic via `hatch-vcs`: the version is derived from the latest `v*` git tag. You cut a release by **publishing a GitHub Release**. That creates the tag and triggers the publish workflow. There is no manual `git tag`, `hatch build`, or `twine upload` step.

1. Confirm `main` is green and the validation sign-off for this version is committed (see [Validation](#validation-against-the-prototype)).
2. Publish the release. Use the GitHub web UI (Releases, then "Draft a new release", create a new tag like `v1.0.0`, then "Publish release"), or the CLI:

   ```bash
   gh release create v1.0.0 --target main --generate-notes
   ```

   The tag name is the version. `v1.0.0` publishes `1.0.0`.
3. The `Release` workflow (`.github/workflows/release.yml`) runs on the `release: published` event. It builds the wheel and sdist with `hatch build`, runs `twine check` plus a wheel-install smoke test, then publishes to PyPI.

Publishing uses **PyPI Trusted Publishing** (OIDC), not an API token. The `upload_pypi` job runs in the `pypi` GitHub Environment and requests an `id-token`. There is no `PYPI_API_TOKEN` secret to manage. For this to work, the PyPI project must have a trusted publisher registered for this repository, the `release.yml` workflow, and the `pypi` environment. Set that up once under the PyPI project's Publishing settings.

The same workflow also builds and validates the distribution (but does **not** publish) on every push to `main` and on pull requests, so a broken build surfaces before you cut a release.
