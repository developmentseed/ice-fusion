# Basic run

The minimum to score a PSU-ISM ensemble and produce weights:

```python
import fusion

cfg = fusion.load_config("my_run.yaml")
result = fusion.run(cfg)

fusion.save_weights(result, "weights.csv")
fusion.plot_projection(result, "sle_2100.png")
fusion.save_metadata(result, "run_metadata.json")
```

`weights.csv` looks like:

| member_id | posterior_mean | posterior_sd | point_estimate |
|-----------|----------------|--------------|----------------|
| run_001   | 0.142          | 0.018        | 0.139          |
| run_002   | 0.097          | 0.014        | 0.094          |
| ...       | ...            | ...          | ...            |
