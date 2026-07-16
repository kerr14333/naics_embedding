# Using the embeddings in a hierarchical Bayesian model

The industry embeddings encode "how close two industries are." This note covers how to
feed that into an HBM without shooting yourself in the foot. Two entry points; the second
is the better fit for most uses.

## Don't: raw 1024-d as fixed effects

With 310 industries, dumping the full `industry_vectors.parquet` in as fixed-effect
predictors is p ≫ n — slow MCMC, non-identifiable, and dependent on brutal priors. Reduce
first, or use the kernel form below.

## Option 1 — embeddings as covariates (PCA)

Use `industry_pca.parquet`, and even then only the leading components (~10–20), not all
116.

- **Standardize the columns.** Rows are L2-normalized; columns are not, so z-score each
  `pc###` before modeling.
- PCA components are orthogonal → no collinearity, but unequal variance (see
  `pca_explained_variance.csv`).
- Put a **regularizing prior** on the slopes — horseshoe, or a tight `Normal(0, σ)` on the
  standardized components.

This treats semantics as ordinary covariates. Fine, but it uses "closeness" only
implicitly.

## Option 2 — embeddings as a covariance kernel (recommended)

This is what the embedding is *for*: let industry-level effects **partially pool by
similarity** instead of shrinking to a single global mean.

```
alpha  ~ MVN(0, tau^2 * K)      # industry-level random effects
K      = cosine_similarity.parquet   # 310 x 310
```

`cosine_similarity.parquet` is a Gram matrix of L2-normalized vectors, so it is **PSD by
construction → a valid GP/covariance kernel**. Drop it straight into a PyMC / Stan /
NumPyro GP prior over the 310 industries (a 310×310 covariance is tiny). Similar
industries (Iron & Steel ↔ Foundries) shrink toward each other; distant ones don't.

A **runnable** version of this is in [`examples/hbm_kernel_example.py`](examples/hbm_kernel_example.py)
(fits prose-tier industries against a synthetic outcome; swap in your real likelihood).

### PyMC sketch

```python
import numpy as np, pandas as pd, pymc as pm

K = pd.read_parquet("out/cosine_similarity.parquet")
codes = K.index.to_list()
Kv = K.to_numpy()
Kv = 0.5 * (Kv + Kv.T)                     # symmetrize
Kv += 1e-6 * np.eye(len(Kv))               # jitter for a stable Cholesky

with pm.Model() as m:
    tau = pm.HalfNormal("tau", 1.0)
    L = np.linalg.cholesky(Kv)
    z = pm.Normal("z", 0, 1, shape=len(Kv))
    alpha = pm.Deterministic("alpha", tau * (L @ z))   # industry effects
    # ... map alpha[industry_index] into your likelihood ...
```

If you'd rather control how fast similarity decays, exponentiate a distance instead of
using raw cosine: `K = exp(-(1 - cos) / ell)` and put a prior on the lengthscale `ell`
(then sensitivity-check it).

## Caveats that bite specifically in an HBM

1. **Aggregates break exchangeability.** The 47 rows with `tier == "aggregate"` (Total
   Nonfarm, Goods Producing, Durable Goods, supersectors, government) are roll-ups of
   other rows. Include an aggregate *and* its children and you double-count the same
   employment. Filter to `tier == "prose"` for a clean exchangeable set, or model the
   aggregates as a separate level. In the distance heatmap these rows show up as a dark
   band — far from everything — because their titles are generic.

2. **A discrete hierarchy already exists.** `supersector_code` gives a classic nested
   grouping (industry ⊂ supersector). You can (a) use it as a discrete group level, (b)
   use the embedding kernel as a continuous replacement, or (c) combine: supersector
   grouping with the kernel operating within.

3. **Which "closeness" is authoritative.** Full-vector cosine (`cosine_similarity.parquet`)
   is the reference. Cosine in PCA-reduced space is *not* equivalent unless all components
   are kept — if you build the kernel, build it from the full-vector cosine matrix, not
   from a truncated PCA.

## Which file for which use

| Use | File |
|---|---|
| Covariates / feature matrix | `industry_pca.parquet` (reduce further before modeling) |
| GP / covariance kernel | `cosine_similarity.parquet` |
| Row quality / filtering | `tier` column in `meta.parquet` / the matrices |
| Discrete grouping level | `supersector_code` in `meta.parquet` |
| Join key | `industry_code` (8-char string — keep the leading zeros) |
