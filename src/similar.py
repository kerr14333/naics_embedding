"""Nearest-industry lookup over the embeddings -- also the sanity check.

Vectors are L2-normalized, so cosine similarity == vecs @ vecs.T.

Usage:
  uv run python src/similar.py                 # run built-in sanity probes
  uv run python src/similar.py 31331100        # neighbors of an industry_code
  uv run python src/similar.py "coal mining"   # neighbors by title substring
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "out"

# A few probes whose neighbors we can eyeball for sanity.
PROBES = ["10212100", "31331100", "20236100", "55520000"]


def load():
    vecs = np.load(OUT / "embeddings.npy")
    meta = pd.read_parquet(OUT / "meta.parquet").reset_index(drop=True)
    return vecs, meta


def resolve(meta: pd.DataFrame, query: str) -> int | None:
    hit = meta.index[meta["industry_code"] == query]
    if len(hit):
        return int(hit[0])
    mask = meta["industry_name"].str.contains(query, case=False, na=False, regex=False)
    return int(meta.index[mask][0]) if mask.any() else None


def neighbors(vecs, meta, i: int, k: int = 6) -> pd.DataFrame:
    sims = vecs @ vecs[i]
    # Fix A: the 45 CES roll-ups (no NAICS match) are hubs -- exclude them from
    # the neighbor pool. They are aggregates, not real industries. You can still
    # query one; it just won't be returned as anyone's neighbor.
    is_aggregate = meta["naics"].isna().to_numpy()
    sims = np.where(is_aggregate, -np.inf, sims)
    order = np.argsort(-sims)[: k + 1]  # +1 to drop self
    order = [j for j in order if j != i and np.isfinite(sims[j])][:k]
    return pd.DataFrame({
        "sim": sims[order].round(3),
        "industry_code": meta.loc[order, "industry_code"].values,
        "industry_name": meta.loc[order, "industry_name"].values,
    })


def show(vecs, meta, query: str, k: int = 6) -> None:
    i = resolve(meta, query)
    if i is None:
        print(f"no match for {query!r}")
        return
    r = meta.loc[i]
    print(f"\n[{r.industry_code}] {r.industry_name}  (supersector: {r.supersector_name})")
    print(neighbors(vecs, meta, i, k).to_string(index=False))


def main() -> None:
    vecs, meta = load()
    queries = sys.argv[1:] or PROBES
    for q in queries:
        show(vecs, meta, q)


if __name__ == "__main__":
    main()
