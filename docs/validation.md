# Validation against the prototype

Before each release, FUSION is shown to reproduce the original `full_model.py` prototype on a frozen reference ensemble. This is a **manual, human-gated** process — not part of CI.

The harness lives in Sara's [`PSUISM_HBM_V1`](https://github.com/sc-peters/PSUISM_HBM_V1) repo, alongside `full_model.py`, not in this package — different audience (Max + Sara, not package users).

## How FUSION makes this possible

- The 20,000-point subsample is configurable via `inference.subsample.size` and `inference.subsample.seed`. The same seed produces the same indices in both FUSION and the patched prototype, so RNG-call order is no longer a source of false diffs.
- `inference.draws`, `tune`, `chains`, `target_accept` are exposed in config so the harness can pin them to the prototype's values.
- Every FUSION run records its seed, config, and obs version in `run_metadata.json`.

## Process

1. **Max runs the harness first** in `PSUISM_HBM_V1`. Walks through the layered diff: bit-exact equality on data loading and scoring; tight numerical tolerance on PyMC posterior summaries (posterior means and quantiles). Fixes any material divergence and writes a short report.
2. **Sara reviews and signs off.** Side-by-side outputs and Max's report go to Sara, who owns the science. Sara's sign-off — not a green CI run — releases v1.

Validation is rerun any time the metric or PyMC model changes.
