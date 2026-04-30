# Usage

## One-shot

```python
import fusion
result = fusion.run(fusion.load_config("my_run.yaml"))
```

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

- **Weights CSV** — one row per ensemble member; both Bayesian posterior (with uncertainty) and the plug-in point estimate.
- **SLE projection** — weighted distribution of sea-level contribution in 2100.
- **`run_metadata.json`** — version, resolved config, obs version, seeds, file hashes.
