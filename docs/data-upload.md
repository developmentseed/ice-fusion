# Data upload to Source.Coop

This page walks through uploading observation bundles to the [Source.Coop](https://source.coop/<org>/<product>) repository using the AWS CLI, starting from scratch.

> Adapted from <https://raw.githubusercontent.com/developmentseed/astera-demogorgn/refs/heads/main/docs/data-upload.md>.

## Background

Source.Coop is a utility for hosting open datasets that provides a public data catalog and standardised access. The data itself physically lives on an Amazon Web Services S3 bucket; the following upload instructions explain how to get started with the AWS CLI in order to upload data to the public repository.

### What gets uploaded (and what doesn't)

A run consumes several inputs, but **only the observation bundles are uploaded to Source.Coop**:

| Input | Where it comes from | Upload here? |
|-------|---------------------|--------------|
| Observation bundles (elevation + velocity NetCDFs) | Downloaded from Source.Coop at runtime (`src/fusion/data/obs.py`) | **Yes, this page** |
| PSU-ISM ensemble (model runs) | Supplied locally by each user via `ensemble.path` in their config (`src/fusion/data/ensemble.py`) | No (never uploaded; stays on the user's disk) |
| Region basins (`imbie_basins`) | Fetched live from `xopr.get_antarctic_regions` (MEaSUREs NSIDC-0709 v2) | No (no file involved) |
| Target grid (`obs_8km`) | Positional 761×761 shape check only | No (no file involved) |

Uploading the obs bundle is therefore all that's needed to make a published version usable; the ensemble is each user's own input and is never published here.

### Repository and version naming

The library reads from a **fixed product slot**: `SOURCE_BUCKET_URL` in `src/fusion/data/obs.py` points at `s3://us-west-2.opendata.source.coop/<org>/fusion-obs`. Throughout this section **`<product>` is `fusion-obs`**. Substitute it (and your `<org>`) consistently, and keep both in sync with `obs.py`.

Bundles are versioned: each upload goes under a `<version>` prefix that callers select via `ObservationsConfig(version=...)`. **Name the version with an ISO date (`YYYY-MM-DD`)**, e.g. `2026-04-30`, so versions sort chronologically and read unambiguously. `version` is a free-form string in the code, so this is a convention rather than an enforced format; apply it to every upload.

---

## Choose an upload method

Source Cooperative accepts uploads two ways. Either way, the files must follow the [bundle layout](#bundle-layout).

- **Option 1, the web UI (easiest).** No credentials or command line needed. Best for a small or one-time upload. Covered just below.
- **Option 2, the AWS CLI.** Best for larger uploads, and for repeatable or scripted delivery. Covered from [Install the AWS CLI](#install-the-aws-cli) onward.

## Option 1: Upload in the web UI (easiest)

For a small or one-time upload you can go straight through the browser, with no AWS CLI or credentials. The [Source.Coop upload docs](https://docs.source.coop/data-upload#option-1-upload-directly-in-the-ui-easiest) are the canonical reference.

First stage the files locally in the [bundle layout](#bundle-layout) below. That is a `<version>/` directory containing `elevation/` and `velocity/`. Then:

1. Go to the product page: `https://source.coop/<org>/fusion-obs`.
2. On the "Product Contents" card, click the lock icon in its top-right corner to open the dropdown.
3. Select "Edit Mode".
4. Add the files. Click "Upload Directory" to add the whole `<version>/` tree at once (recommended), or drag and drop onto the card, or "Upload Files" for individual files.
5. Files upload automatically once selected.

The UI documents no size or count limit. For large bundles, or for uploads you need to repeat, prefer the AWS CLI below.

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

What `fusion.data.obs.load_observations` actually requires (everything else is convention):

- **Stream directories named exactly `elevation/` and `velocity/`.** These names are hardcoded (`obs.py`); the stream is *not* inferred from filenames.
- **Each file is a `*.nc` whose name contains a four-digit year.** The year is parsed as the first four-digit run in the filename (`elev_antarctica_elevation_2015.nc` → `2015`); the rest of the filename is ignored. The `elev_…` / `vel_…` prefixes shown above are a recommended convention, not a requirement.
- **The variables inside must be named exactly** `height` and `absolute_elevation_rmse` (elevation) and `VX`, `VY`, `ERRX`, `ERRY` (velocity). The v1 metric selects them by name, so a mismatch raises `KeyError`.

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

After upload, callers select the new bundle by setting `ObservationsConfig(version="<version>")` in their config, or by passing `observations.version: "<version>"` in their YAML.
