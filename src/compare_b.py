"""Fix B experiment: embed description-only sentences and compare neighbors
against the current title+description embeddings (Fix A already applied).

Matched industries -> Census description alone (drop the title, killing the
"...Manufacturing" word-echo). Unmatched aggregates -> title fallback (unchanged),
and they stay excluded from the neighbor pool either way.

Saves B vectors to out/embeddings_desc.npy and prints A-vs-B neighbors for the
sanity probes.

Run: uv run python src/compare_b.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "out"
MODEL = "BAAI/bge-large-en-v1.5"
PROBES = ["10212100", "31331100", "20236100", "55520000"]


def desc_only(row) -> str:
    # Census 'description' isn't stored in corpus.parquet; recover it from the
    # title+description sentence: everything after the first ". " if matched.
    if pd.isna(row["naics"]):
        return row["industry_name"]
    s = row["sentence"]
    return s.split(". ", 1)[1] if ". " in s else s


def neighbors(vecs, meta, i, k=6):
    sims = vecs @ vecs[i]
    sims = np.where(meta["naics"].isna().to_numpy(), -np.inf, sims)
    order = [j for j in np.argsort(-sims) if j != i and np.isfinite(sims[j])][:k]
    return [(round(float(sims[j]), 3), meta.loc[j, "industry_name"]) for j in order]


def main() -> None:
    meta = pd.read_parquet(OUT / "meta.parquet").reset_index(drop=True)
    vecs_a = np.load(OUT / "embeddings.npy")

    sents_b = meta.apply(desc_only, axis=1).tolist()
    model = SentenceTransformer(MODEL)
    vecs_b = model.encode(sents_b, batch_size=32, normalize_embeddings=True,
                          show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    np.save(OUT / "embeddings_desc.npy", vecs_b)

    for code in PROBES:
        i = int(meta.index[meta["industry_code"] == code][0])
        print(f"\n=== [{code}] {meta.loc[i,'industry_name']} ===")
        na, nb = neighbors(vecs_a, meta, i), neighbors(vecs_b, meta, i)
        print(f"{'A: title+desc':<52}   B: desc-only")
        for (sa, na_), (sb, nb_) in zip(na, nb):
            print(f"{sa:.3f} {na_[:44]:<46}   {sb:.3f} {nb_[:44]}")


if __name__ == "__main__":
    main()
