# Validation against the prototype

Before each release, FUSION is shown to reproduce the original `full_model.py` prototype on a frozen reference ensemble. This is a **manual, human-gated** process. It is not part of CI.

The harness lives in `validation/` in this repo, alongside a pinned copy of the prototype it's compared against. The science owner maintains the canonical prototype upstream at [`PSUISM_HBM_V1`](https://github.com/sc-peters/PSUISM_HBM_V1); `validation/baseline/full_model.py` here is a frozen fork the harness imports, with minimal env-var-override patches documented in its README.

## How FUSION makes this possible

- The 20,000-point subsample is configurable via `inference.subsample.size` and `inference.subsample.seed`. The same seed produces the same indices in both FUSION and the patched baseline, so RNG-call order is no longer a source of false diffs.
- `inference.draws`, `tune`, `chains`, `target_accept` are exposed in config so the harness can pin them to the prototype's values.
- `fusion.pipeline.plug_in_weights` returns the per-member raw log-likelihood alongside the softmax weight, exposing the same value the prototype writes as `log_likelihood` in `model_weights_table.csv` for bit-exact comparison.
- Every FUSION run records its seed, config, and obs version in `run_metadata.json`.

## Process

1. **Drop the reference inputs into `validation/data/`** per the layout in `validation/baseline/README.md`. The directory is gitignored.
2. **Run the harness:** `uv run python -m validation.compare`. It runs both stacks against the same inputs, diffs at three layers (bit-exact prepared arrays → bit-exact per-pixel log-likelihood → `rtol=1e-3` posterior summaries), and writes `validation/reports/<YYYY-MM-DD>.md` with sign-off slots.
3. **The science owner reviews and signs off.** Side-by-side outputs and the report go to whoever owns the science. Their sign-off releases v1, not a green CI run.

Validation is rerun any time the metric or PyMC model changes, or when the upstream prototype is refreshed into `validation/baseline/`.
