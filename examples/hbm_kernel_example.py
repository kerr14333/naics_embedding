"""Runnable example: use the industry cosine matrix as a covariance kernel in a
hierarchical Bayesian model (PyMC).

This is the "Option 2" recipe from MODELING.md. It fits industry-level random
effects that partially pool by embedding similarity -- similar industries shrink
toward each other rather than toward a single global mean.

The likelihood here is a stand-in (a synthetic Normal outcome per industry) so the
script runs on its own. Replace the `outcome` / likelihood block with your real
observation model (e.g. state x industry employment growth, with alpha indexing the
industry level).

PyMC is NOT a core project dependency (it's heavy and only needed for this example):
    uv add pymc        # or: pip install pymc

Run: uv run python examples/hbm_kernel_example.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "out"


def load_kernel(prose_only: bool = True):
    """Return (codes, K) -- the cosine kernel, optionally restricted to prose-tier
    industries so aggregate roll-ups don't break exchangeability (MODELING.md caveat 1)."""
    K = pd.read_parquet(OUT / "cosine_similarity.parquet")
    meta = pd.read_parquet(OUT / "meta.parquet").set_index("industry_code")
    if prose_only:
        keep = meta.index[meta["tier"] == "prose"]
        K = K.loc[keep, keep]
    Kv = K.to_numpy()
    Kv = 0.5 * (Kv + Kv.T)                 # symmetrize numerical drift
    Kv += 1e-6 * np.eye(len(Kv))           # jitter for a stable Cholesky
    return K.index.to_list(), Kv


def main() -> None:
    try:
        import pymc as pm
    except ModuleNotFoundError:
        raise SystemExit("PyMC not installed. Run:  uv add pymc")

    codes, Kv = load_kernel(prose_only=True)
    n = len(codes)
    L = np.linalg.cholesky(Kv)

    # --- synthetic outcome, replace with your real data ---
    rng = np.random.default_rng(0)
    true_alpha = L @ rng.standard_normal(n) * 0.8
    outcome = true_alpha + rng.normal(0, 0.3, size=n)  # one obs per industry
    # ------------------------------------------------------

    with pm.Model() as model:
        tau = pm.HalfNormal("tau", 1.0)            # kernel amplitude
        sigma = pm.HalfNormal("sigma", 1.0)        # observation noise
        z = pm.Normal("z", 0, 1, shape=n)
        alpha = pm.Deterministic("alpha", tau * (L @ z))   # industry effects ~ MVN(0, tau^2 K)
        pm.Normal("y", mu=alpha, sigma=sigma, observed=outcome)

        idata = pm.sample(500, tune=500, chains=2, target_accept=0.9,
                          random_seed=0, progressbar=True)

    summ = pm.summary(idata, var_names=["tau", "sigma"])
    print(summ)
    print(f"\nfit {n} prose-tier industries; alpha is the partially-pooled effect per code.")
    print("map alpha back to industries via `codes` (row order preserved).")


if __name__ == "__main__":
    main()
