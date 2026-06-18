# Terminology

A glossary of the acronyms and terms used across these docs. Acronyms also carry their expansion as a tooltip throughout the site: hover over an acronym such as SLE or MCMC on any page to see what it stands for.

## Acronyms

| Acronym | Stands for | What it means here |
|---------|------------|--------------------|
| FUSION | the project name | The `ice-fusion` library (PyPI name `ice-fusion`, import name `fusion`). It scores an ice-sheet model ensemble against observations and weights each member's projection by historical skill. |
| SLE | sea-level equivalent | The contribution an ice volume would make to global mean sea level. The projection is an SLE distribution at the target year. |
| PSU-ISM | Penn State University Ice Sheet Model | The ice-sheet model whose ensemble members FUSION scores. |
| HBM | Hierarchical Bayesian Model | The structure of the inference: per-stream parameters share priors and combine into member weights. The prototype is `PSUISM_HBM_V1`. |
| MCMC | Markov chain Monte Carlo | The sampling method behind the posterior. Member weights are estimated by MCMC, so they are only trustworthy once the chains converge. |
| NUTS | No-U-Turn Sampler | The specific MCMC algorithm PyMC runs. |
| ESS | effective sample size | How many effectively-independent posterior draws inform a quantity. Reported as `min_ess_bulk` and `min_ess_tail` in the diagnostics. |
| PyMC | (library name) | The Python probabilistic-programming library that builds and samples the model. |
| VAF | volume above flotation | The physically correct basis for an SLE reduction. The v1 projection is a placeholder and does not yet compute it. |
| RMSE | root-mean-square error | The elevation uncertainty field is `absolute_elevation_rmse`. |
| NetCDF | Network Common Data Form | The on-disk format for the gridded observation and ensemble inputs (`*.nc`). |
| HPC | high-performance computing | Cluster login nodes are a common reason to relocate the cache with `FUSION_CACHE`. |
| RNG | random number generator | Matching RNG-call order between FUSION and the baseline is what keeps subsampling reproducible. |
| NSIDC | National Snow and Ice Data Center | Source of the region basins (`MEaSUREs NSIDC-0709 v2`). |
| MEaSUREs | Making Earth System Data Records for Use in Research Environments | The NASA program providing the region basins. |
| IMBIE | Ice sheet Mass Balance Inter-comparison Exercise | The basin definition behind the `imbie_basins` regions option. |
| AWS | Amazon Web Services | Source.Coop stores the observation bundles on AWS S3. |
| S3 | Amazon Simple Storage Service | The object store the observation bundles live in (`s3://...`). |
| CLI | command-line interface | The AWS CLI is one of the two upload methods for the obs bundle. |
| API | application programming interface | The public Python API is whatever `fusion/__init__.py` re-exports. |
| OIDC | OpenID Connect | The mechanism behind PyPI Trusted Publishing; no API token is stored. |
| PyPI | Python Package Index | Where `ice-fusion` is published. |
| YAML | YAML Ain't Markup Language | The config file format [`load_config`][fusion.load_config] reads. |
| ISO | International Organization for Standardization | Obs bundle versions use ISO dates (`YYYY-MM-DD`). |

!!! note "CI is overloaded"
    "CI" means two different things in these docs, so it is left out of the hover tooltips. In the context of results it is the **credible interval** (e.g. the 5–95% interval on the projection). In the context of development and releases it is **continuous integration** (the GitHub Actions checks). The surrounding text always makes clear which one is meant.

## Other terms

| Term | What it means |
|------|---------------|
| R-hat (Gelman–Rubin) | A convergence diagnostic comparing between-chain and within-chain variance. A value above 1.01 means the chains disagree. Reported as `max_rhat`. |
| Divergence | A transition the sampler could not integrate accurately. Any divergences (`n_divergences > 0`) can bias the posterior, not just add noise. |
| Posterior mean | The headline member weight: the mean of a member's Bayesian weight `w` over the posterior. Weights sum to 1 across members. |
| Point estimate | The prototype's plug-in weight: a softmax of per-member log-likelihood at the posterior-mean parameters. Used as a cross-check on the posterior mean. |
| Credible interval | A Bayesian interval on a quantity. The projection is reported as a median with a 5–95% credible interval. |
| Stream | An observation type fed to the metric: elevation (thickness rate) or velocity. The two streams combine via `stream_weights`. |
| Plug-in weights | Deterministic weights computed by plugging the posterior-mean parameters into the likelihood, rather than averaging over the posterior. |
| Bit-exactness | Reproducing the prototype's arrays exactly (`np.array_equal`), not just within a tolerance. The validation harness checks this at Layers 1 and 2. |
| Science territory | Modules that port the prototype's science verbatim (`inference/model.py`, `data/prepare.py`). Their pinned defaults must not change without re-validating. |
