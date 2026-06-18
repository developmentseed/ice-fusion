# Pinned prototype (oracle)

Frozen fork of Sara Peters' `full_model.py` for use by the validation
harness in `../compare.py`. The harness imports this file rather than the
canonical upstream so any modifications needed to drive a side-by-side
comparison are version-controlled and diffable against upstream.

## Upstream

Source: <https://github.com/sc-peters/PSUISM_HBM_V1/tree/main/push%204_13_26>

When the upstream prototype changes, refresh this directory:

1. Re-download `full_model.py` from the upstream URL above.
2. Re-apply the patches listed below (or re-port them on top of the new
   file if upstream has moved them).
3. Run `../compare.py` and update the report.
4. Commit baseline + report together so the diff is reviewable.

## Patches applied vs upstream

All marked inline with `# PATCH (validation/baseline): ...` comments.

- **`import os`** added at the top of the imports block.
- **Path constants** in `load_and_prepare_data()` (`OBS_THICKNESS_DIR`,
  `OBS_VELOCITY_DIR`, `MODEL_DIR`) now read from environment variables —
  `FUSION_OBS_THICKNESS_DIR`, `FUSION_OBS_VELOCITY_DIR`, `FUSION_MODEL_DIR` —
  with the upstream hardcoded paths preserved as defaults.
- **Subsample knobs** in `prepare_for_inference()` (`MAX_POINTS = 20000`,
  the seed `42`) now read from `FUSION_SUBSAMPLE_SIZE` and
  `FUSION_SUBSAMPLE_SEED` with the upstream values as defaults.
- **`pm.sample` seeding** in `run_mcmc()` — upstream omits `random_seed`,
  leaving PyMC to choose nondeterministically. The harness needs a fixed
  seed so the prototype's MCMC matches ice-fusion's seeded run; reads
  from `FUSION_PYMC_SEED` (default 42).

The defaults are byte-for-byte identical to upstream, so running the file
without the env vars set produces the same behaviour as the canonical
prototype in its upstream environment.
