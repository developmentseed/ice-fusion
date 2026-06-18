# Contributing

Welcome, and thank you for picking this up. This page is the handoff guide for taking over `ice-fusion`.

The package itself is on PyPI as `ice-fusion`; the import name is `fusion`. Source lives at [developmentseed/ice-fusion](https://github.com/developmentseed/ice-fusion) on GitHub, but will be transferred.

## What you're inheriting

`ice-fusion` ports Sara Peters' [`PSUISM_HBM_V1`](https://github.com/sc-peters/PSUISM_HBM_V1) prototype (`full_model.py`) into a packaged library. The parts that are "Science territory" — the metric definition, the priors, the per-stream weights — are deliberately marked as such in the source:

- `src/fusion/inference/model.py` — the hierarchical Bayesian model, ported from `build_model_proposal` in the prototype.
- `src/fusion/data/prepare.py` — the rate-of-change preparation that builds the inputs the PyMC model consumes.

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

## Validation against the prototype

This is the gate that releases v1. See [Validation](validation.md) for the full process. In short:

1. Drop reference inputs into `validation/data/` (gitignored — see `validation/baseline/README.md` for the layout).
2. Run the harness:
   ```bash
   uv run python -m validation.compare
   ```
3. The harness writes `validation/reports/<YYYY-MM-DD>.md`. The report contains a sign-off block with two checkboxes (Max + Sara). Only your sign-off — not a green CI run — releases a version.

The comparison runs at three layers:

- **Layer 1** — prepared arrays (`y_obs`, `sigma_obs`, `F`, `speed`, `n_dhdt`, `n_vel`): bit-exact, `np.array_equal`.
- **Layer 2** — per-member plug-in log-likelihood at a canonical posterior mean: bit-exact.
- **Layer 3** — posterior summaries (`sigma_base_*`, `beta_*`, `w`): `rtol=1e-3`.

Re-run validation any time the metric or PyMC model changes, and any time the upstream prototype is refreshed into `validation/baseline/`.

## Refreshing the baseline from upstream

You own the canonical prototype upstream at `sc-peters/PSUISM_HBM_V1`. When the upstream version changes meaningfully, refresh `validation/baseline/full_model.py` so the harness compares against current science:

1. Re-download `full_model.py` from the upstream repo.
2. Re-apply the env-var-override patches listed in `validation/baseline/README.md` (paths, subsample seed, subsample size). They're marked inline with `# PATCH (validation/baseline): ...` comments.
3. Run `uv run python -m validation.compare`.
4. Commit the refreshed baseline and the new validation report together so the diff is reviewable.

If the refreshed baseline breaks Layer 1 or Layer 2, the port needs an update — that's the signal to change `src/fusion/`.

## Releasing

Versioning is automatic via `hatch-vcs`: the next git tag is the version.

```bash
git tag v1.0.0
git push --tags
```

CI builds and publishes to PyPI on tag push (assuming the `PYPI_API_TOKEN` secret is configured — set this in GitHub repo settings if it isn't already).

---

# Data Upload to Source.Coop

This section walks through uploading observation bundles to the [Source.Coop](https://source.coop/<org>/<product>) repository using the AWS CLI, starting from scratch.

> Adapted from <https://raw.githubusercontent.com/developmentseed/astera-demogorgn/refs/heads/main/docs/data-upload.md>.

## Background

Source.Coop is a utility for hosting open datasets that provides a public data catalog and standardised access. The data itself physically lives on an Amazon Web Services S3 bucket; the following upload instructions explain how to get started with the AWS CLI in order to upload data to the public repository.

### What gets uploaded (and what doesn't)

A run consumes several inputs, but **only the observation bundles are uploaded to Source.Coop**:

| Input | Where it comes from | Upload here? |
|-------|---------------------|--------------|
| Observation bundles (elevation + velocity NetCDFs) | Downloaded from Source.Coop at runtime (`src/fusion/data/obs.py`) | **Yes — this section** |
| PSU-ISM ensemble (model runs) | Supplied locally by each user via `ensemble.path` in their config (`src/fusion/data/ensemble.py`) | No — never uploaded; stays on the user's disk |
| Region basins (`imbie_basins`) | Fetched live from `xopr.get_antarctic_regions` (MEaSUREs NSIDC-0709 v2) | No — no file involved |
| Target grid (`obs_8km`) | Positional 761×761 shape check only | No — no file involved |

Uploading the obs bundle is therefore all that's needed to make a published version usable; the ensemble is each user's own input and is never published here.

### Repository and version naming

The library reads from a **fixed product slot**: `SOURCE_BUCKET_URL` in `src/fusion/data/obs.py` points at `s3://us-west-2.opendata.source.coop/<org>/fusion-obs`. Throughout this section **`<product>` is `fusion-obs`** — substitute it (and your `<org>`) consistently, and keep both in sync with `obs.py`.

Bundles are versioned: each upload goes under a `<version>` prefix that callers select via `ObservationsConfig(version=...)`. **Name the version with an ISO date (`YYYY-MM-DD`)** — e.g. `2026-04-30` — so versions sort chronologically and read unambiguously. `version` is a free-form string in the code, so this is a convention rather than an enforced format; apply it to every upload.

---

## Install the AWS CLI

=== "macOS"

    ```bash
    brew install awscli
    ```

=== "Linux"

    ```bash
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    sudo ./aws/install
    ```

=== "Windows"

    Download and run the [official MSI installer](https://awscli.amazonaws.com/AWSCLIV2.msi).

Verify the installation:

```bash
aws --version
```

---

## Get credentials from Source

Source provides S3-compatible credentials scoped to a specific repository. To retrieve them:

1. Go to the repository page: [source.coop/<org>/<product>](https://source.coop/<org>/<product>)
2. Click the **Lock Icon** to the right of "Product Contents" (you must be logged in and have write access)
3. Click **View Credentials**
4. Click **Environment Variables**
5. Copy and paste the content under "For terminal/shell usage" into your CLI

---

## Verify access

List the repository prefix to confirm your credentials work:

```bash
aws s3 ls s3://us-west-2.opendata.source.coop/<org>/<product>/
```

You should see one prefix per uploaded version (e.g. `2026-04-30/`). A `NoCredentialsError` or `AccessDenied` response means the credentials or profile configuration need rechecking.

---

## Upload a bundle

A bundle is a versioned directory tree containing one NetCDF per year per stream.

### Bundle layout

```
<version>/
    elevation/
        elev_antarctica_elevation_2015.nc   # vars: height, absolute_elevation_rmse
        elev_antarctica_elevation_2016.nc
        ...
    velocity/
        vel_Antarctica_ice_velocity_2014.nc # vars: VX, VY, ERRX, ERRY
        vel_Antarctica_ice_velocity_2015.nc
        ...
```

What `fusion.data.obs.load_observations` actually requires — everything else is convention:

- **Stream directories named exactly `elevation/` and `velocity/`.** These names are hardcoded (`obs.py`); the stream is *not* inferred from filenames.
- **Each file is a `*.nc` whose name contains a four-digit year.** The year is parsed as the first four-digit run in the filename (`elev_antarctica_elevation_2015.nc` → `2015`); the rest of the filename is ignored. The `elev_…` / `vel_…` prefixes shown above are a recommended convention, not a requirement.
- **The variables inside must be named exactly** `height` and `absolute_elevation_rmse` (elevation) and `VX`, `VY`, `ERRX`, `ERRY` (velocity) — the v1 metric selects them by name, so a mismatch raises `KeyError`.

Keep to the filenames shown for consistency with existing bundles, but only the three rules above are load-bearing.

### Upload a whole bundle

Use `aws s3 sync` for the typical case. Stage the bundle locally as `<version>/elevation/...` and `<version>/velocity/...`, then preview with `--dryrun` before uploading:

```bash
aws s3 sync ./2026-04-30/ \
    s3://us-west-2.opendata.source.coop/<org>/<product>/2026-04-30/ \
    --exclude "*" \
    --include "elevation/*.nc" \
    --include "velocity/*.nc" \
    --dryrun
```

The `--exclude "*"` + `--include` pair is a strict allowlist: only the NetCDF files under `elevation/` and `velocity/` will be uploaded, and incidental files like `.DS_Store`, `.ipynb_checkpoints/`, or editor swap files are filtered out. Inspect the `(dryrun)` lines, then re-run without `--dryrun` to perform the upload.

### Upload or replace a single file

```bash
aws s3 cp elev_antarctica_elevation_2015.nc \
    s3://us-west-2.opendata.source.coop/<org>/<product>/2026-04-30/elevation/elev_antarctica_elevation_2015.nc
```

### Path layout reference

| File type | S3 path |
|-----------|---------|
| Elevation, year `YYYY` | `s3://us-west-2.opendata.source.coop/<org>/<product>/<version>/elevation/elev_antarctica_elevation_<YYYY>.nc` |
| Velocity, year `YYYY` | `s3://us-west-2.opendata.source.coop/<org>/<product>/<version>/velocity/vel_Antarctica_ice_velocity_<YYYY>.nc` |

After upload, callers select the new bundle by setting `ObservationsConfig(version="<version>")` in their config — or by passing `observations.version: "<version>"` in their YAML.
