# Interpreting results

A run produces three things. A per-member **weights table**, an SLE **projection**, and a set of **convergence diagnostics**. The weights and the projection are only meaningful if the sampler that produced them converged. So start with the diagnostics.

## Did the run converge?

The member weights come from a posterior estimated by MCMC (NUTS). If the chains did not converge, the weights are unreliable. So is every number downstream of them, however plausible they look.

`fusion.run` checks this for you. It emits a `fusion.ConvergenceWarning` when something is off:

```text
ConvergenceWarning: max R-hat 1.043 exceeds 1.01. Chains have not converged;
increase inference.tune/draws.
```

You can also inspect the numbers directly. Every `Result` carries them. `sampler_diagnostics` works on any trace:

```python
import fusion

result = fusion.run(cfg)
result.diagnostics
# {'max_rhat': 1.002, 'min_ess_bulk': 1840.0, 'min_ess_tail': 1610.0, 'n_divergences': 0.0}

fusion.sampler_diagnostics(result.trace)   # same dict, callable on any InferenceData
```

### What the numbers mean

| Key | What it measures | Healthy |
|-----|------------------|---------|
| `max_rhat` | Worst-case Gelman–Rubin R-hat across the four sampled parameters (`sigma_base_thick`, `sigma_base_vel`, `beta_thick`, `beta_vel`). It compares between-chain and within-chain variance. A value above 1.01 means the chains disagree. | ≤ 1.01 |
| `min_ess_bulk` | Smallest **bulk** effective sample size. This is how many effectively-independent draws inform the center of the posterior. | hundreds or more |
| `min_ess_tail` | Smallest **tail** effective sample size. Same idea, for the distribution's tails. The projection's credible interval rests on these. | hundreds or more |
| `n_divergences` | Count of divergent transitions. A divergence means the sampler hit a region it could not integrate accurately. The posterior may be **biased**, not just noisy. | 0 |

These four diagnostics are also written into `run_metadata.json`. A saved run therefore records whether it converged.

### What to do when a check fails

| Symptom | Fix |
|---------|-----|
| `max_rhat` above 1.01, or low `min_ess_*` | Raise `inference.tune` and/or `inference.draws`, then rerun. |
| `n_divergences` above 0 | Raise `inference.target_accept` (e.g. `0.95` to `0.99`). If divergences remain, also raise `tune`. |
| Big gap between `posterior_mean` and `point_estimate` in the weights table | This is a symptom of poor convergence. Check the diagnostics above before trusting the weights. |

!!! note "This is not the same as weight stability"
    `fusion.sampler_diagnostics` asks one question. Did the MCMC chains converge? A separate helper, `fusion.weight_stability_across_seeds`, asks a different one. How sensitive are the weights to which 20k pixels were subsampled? It answers by comparing several runs with different `inference.subsample.seed` values. A run can converge cleanly and still have subsample-sensitive weights, or the reverse. Check both before reporting.

## The weights table

`fusion.save_weights(result, "weights.csv")` writes one row per ensemble member:

| Column | Meaning |
|--------|---------|
| `member_id` | Member label (e.g. `run07`), parsed from the ensemble filename. |
| `posterior_mean` | Mean of the member's Bayesian weight `w` over the posterior. This is the headline skill weight. It sums to 1 across members. |
| `posterior_sd` | Spread of `w` across posterior samples. This is the uncertainty on that weight. |
| `point_estimate` | The prototype's plug-in weight: softmax of the per-member log-likelihood at the posterior-mean parameters. It is deterministic, and close to `posterior_mean` when the chains converged. |
| `point_estimate_loglik` | The N-scaled per-member Gaussian log-likelihood, before the softmax. This is the raw goodness-of-fit. Use it to compare relative fit between members. It is not a weight. |

**Which to use:** report `posterior_mean` with `posterior_sd` as its uncertainty. Use `point_estimate` as a cross-check, per the table above.

## The projection

`result.projection` is the weighted sea-level-equivalent (SLE) distribution at `projection.target_year`. It holds one value per posterior sample. Each member's projected SLE is weighted by its skill weight. Report it as a central estimate plus a credible interval, not a bare mean:

```python
fusion.projection_summary(result.projection)
# {'median': 0.071, 'mean': 0.072, 'sd': 0.011, 'lower': 0.054,
#  'upper': 0.091, 'lower_q': 0.05, 'upper_q': 0.95}

fusion.plot_projection(result, "sle_2100.png")   # histogram with median + 5–95% CI
```

Quote the **median** with the **5–95% credible interval** (`lower` to `upper`).

!!! warning "Projection magnitudes are provisional"
    The member to SLE reduction in v1 is a **placeholder**. It is a mean of ice thickness, not the real volume-above-flotation calculation. Until that lands, treat the projection's shape and relative spread as illustrative. Do not read the absolute SLE magnitude as physically meaningful. The weighting itself, the weights table, is validated and unaffected.
