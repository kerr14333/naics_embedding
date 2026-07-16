"""Fix C experiment: strip the shared NAICS boilerplate prefix
("This industry comprises establishments primarily engaged in ...") before
embedding, to cut the phrase-echo hub. Sentence = title + stripped description.

Compares C neighbors against current A (title + full description).
Run: uv run python src/compare_c.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

OUT = Path(__file__).resolve().parent.parent / "out"
MODEL = "BAAI/bge-large-en-v1.5"
PROBES = ["10212100", "31331100", "20236100", "55520000"]

# Boilerplate openers shared across NAICS descriptions.
BOILER = re.compile(
    r"^This (industry|industry group|sector|subsector) comprises "
    r"establishments (primarily )?(engaged in|known as|that|primarily)\s*",
    re.IGNORECASE,
)


def strip_boiler(text: str) -> str:
    return BOILER.sub("", text).strip()


def c_sentence(row) -> str:
    if pd.isna(row["naics"]):
        return row["industry_name"]
    title, _, desc = row["sentence"].partition(". ")
    return f"{title}. {strip_boiler(desc)}"


def neighbors(vecs, meta, i, k=6):
    sims = vecs @ vecs[i]
    sims = np.where(meta["naics"].isna().to_numpy(), -np.inf, sims)
    order = [j for j in np.argsort(-sims) if j != i and np.isfinite(sims[j])][:k]
    return [(round(float(sims[j]), 3), meta.loc[j, "industry_name"]) for j in order]


def main() -> None:
    meta = pd.read_parquet(OUT / "meta.parquet").reset_index(drop=True)
    vecs_a = np.load(OUT / "embeddings.npy")

    sents_c = meta.apply(c_sentence, axis=1).tolist()
    print("sample stripped:", sents_c[7][:120], "\n")
    model = SentenceTransformer(MODEL)
    vecs_c = model.encode(sents_c, batch_size=32, normalize_embeddings=True,
                          show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    np.save(OUT / "embeddings_stripped.npy", vecs_c)

    for code in PROBES:
        i = int(meta.index[meta["industry_code"] == code][0])
        print(f"\n=== [{code}] {meta.loc[i,'industry_name']} ===")
        print(f"{'A: title+full desc':<52}   C: boilerplate stripped")
        for (sa, na), (sc, nc) in zip(neighbors(vecs_a, meta, i), neighbors(vecs_c, meta, i)):
            print(f"{sa:.3f} {na[:44]:<46}   {sc:.3f} {nc[:44]}")


if __name__ == "__main__":
    main()
