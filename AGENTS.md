# AGENTS.md

This file provides guidance to coding agents (Claude Code, Codex, Cursor, Gemini CLI, etc.) when working with code in this repository.

## What this is

`ice-fusion` (PyPI name `ice-fusion`, import name `fusion`) scores an ensemble of ice-sheet model runs against modern satellite observations over the historical period, then weights each member's future sea-level projection by its historical skill. It is a packaged port of Sara Peters' [`PSUISM_HBM_V1`](https://github.com/sc-peters/PSUISM_HBM_V1) prototype (`full_model.py`).

## Commands

The project uses `uv` (Python 3.12+).

```bash
uv sync --all-extras          # install with dev + docs extras
uv run pytest                 # fast suite (slow PyMC tests excluded by default)
uv run pytest -m slow         # end-to-end tests that exercise the PyMC sampler
uv run pytest tests/test_prepare.py::test_name   # single test
uv run ruff check .           # lint (E, F, I, UP, B, SIM)
uv run ruff format .          # format (line-length 100)
uv run mypy                   # strict, files = src, tests, validation/compare.py
```

Pre-commit runs ruff + ruff-format + codespell on commit/push. mypy is **strict** with `disallow_any_explicit` on — implicit `Any` from untyped scientific libs is handled at the boundary (e.g. pinning a return type after `pm.sample`), not banned project-wide. Untyped third-party deps are listed explicitly under `[[tool.mypy.overrides]]` rather than via a blanket ignore, so a new untyped dependency surfaces as an error and gets a deliberate entry.

## Architecture

The pipeline is four composable steps, each callable on its own and composed by `run`. Defined in `src/fusion/pipeline.py`:

```
load_data → prepare → sample → project
```

- **`load_data`** (`data/obs.py`, `data/ensemble.py`) — fetches the versioned observation bundle from a Source.Coop S3 bucket (cached on disk under `FUSION_CACHE`, default `~/.cache/fusion/<version>/`; first run downloads, later runs are offline) and loads the PSU-ISM ensemble. `_check_grid_compatible` enforces that obs and ensemble share a 761×761 `(y, x)` grid — the v1 metric is purely positional and does **not** compare coord values; native-resolution regridding is a deferred v1.1 feature (raises `NotImplementedError`).
- **`prepare`** (`data/prepare.py`) — finite-differences modelled fields into rate-of-change, aligns obs to model intervals, masks non-finite / above-threshold pixels, and seed-subsamples to 20k pixels per region. Output is `PreparedData`: flat arrays whose first `n_dhdt` entries are thickness-rate obs and the rest are velocity-rate (`vx` then `vy`).
- **`sample`** (`inference/model.py`) — the hierarchical Bayesian PyMC model. Half-normal priors on `sigma_base_*` / `beta_*`; per-pixel uncertainty inflated by `sigma_base * sqrt(1 + beta * speed)`; per-stream log-likelihoods combined with `stream_weights` and registered as a `pm.Potential`; member weights `w` exposed as a deterministic softmax.
- **`project`** (`projection.py`) — applies posterior weights to per-member sea-level-equivalent at the target year.

`run` returns a `Result` (`result.py`) bundling config, data, prepared arrays, trace, weights DataFrame, and projection. The public API is whatever `src/fusion/__init__.py` re-exports.

Config is a pydantic v2 tree (`config.py`) loaded from YAML via `load_config`; validation errors are re-raised as `ValueError` with the path. See `docs/config-reference.md`.

### "Science territory" and bit-exactness

`inference/model.py` and `data/prepare.py` are direct ports of the prototype's science and are marked **"Science territory"** in their module docstrings. The metric definition, priors, per-stream weights (`0.5/0.5`), uncertainty thresholds (`50.0` thickness, `10.0` velocity), and subsample (size `20000`, seed `42`) all default to the prototype's hardcoded values. **Changing these defaults breaks bit-exactness against the validation baseline** — they were surfaced into config so values land in run metadata and v1.1 can evolve them, not so they should be casually changed. Several vectorized rewrites preserve per-row summation order specifically to stay bit-exact with the prototype's loop versions (see `tests/test_inference_vectorised.py`, `tests/test_plug_in_weights.py`).

### Validation harness

`validation/` is a manual side-by-side harness (NOT run in CI) that is the release gate. `validation/baseline/full_model.py` is a pinned, patched fork of the upstream prototype (the **oracle**); `validation/compare.py` runs both stacks and diffs them at three layers:

- **Layer 1** — prepared arrays: bit-exact (`np.array_equal`)
- **Layer 2** — per-member plug-in log-likelihood at a canonical posterior mean: bit-exact
- **Layer 3** — posterior summaries: `rtol=1e-3`

Run with `uv run python -m validation.compare`; it writes `validation/reports/<date>.md` with a human sign-off block. **Only a sign-off — not green CI — releases a version.** Re-run whenever the metric, the PyMC model, or the upstream baseline changes. Refreshing the baseline: re-download upstream, re-apply the `# PATCH (validation/baseline): ...` env-var-override patches (paths, subsample knobs, `pm.sample` seed), re-run, commit baseline + report together. See `docs/contributing.md` and `validation/baseline/README.md`.

`validation/baseline/full_model.py` is **read-only upstream code** — excluded from ruff and mypy. Style/type issues there reflect upstream, not this project; do not "fix" them.

## Conventions and gotchas

- `data/obs.py` and `docs/contributing.md` contain `<org>` / Source bucket **placeholders** pending data upload — don't treat `SOURCE_BUCKET_URL` as final.
- `_sle_per_member` in `pipeline.py` is an explicit **placeholder** (mean of `h`); the real volume-above-flotation → SLE reduction is a scientist-owned open question and is not release-quality.
- Versioning is automatic via `hatch-vcs` (next git tag = version); `src/fusion/_version.py` is generated. Don't hand-edit it.
- `dev-docs/` and `.vscode/*.md` are private internal notes (plans, meeting notes, specs) — useful for context, but never cite them from shipped code or public docs.
- The user commits and pushes manually — propose commits in Conventional Commits style; don't run `git commit`/`git push` unless explicitly asked.
