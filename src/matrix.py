"""Export the industry embeddings as a predictor/design matrix for modeling.

Artifacts (all keyed by industry_code as an 8-char STRING -- leading zeros are
significant, e.g. "05000000"):

  out/industry_vectors.parquet  full 1024-d features: industry_code, tier, emb_0000..
  out/industry_pca.parquet      PCA-reduced features: industry_code, tier, pc001..
  out/industry_pca.csv          CSV mirror of the PCA matrix (light, universal)
  out/pca_explained_variance.csv per-component + cumulative variance
  out/cosine_similarity.parquet 310x310 full-vector cosine matrix (authoritative
                                closeness; PCA-space cosine is NOT equivalent unless
                                all components are kept)

Notes baked in from review:
  - tier column ("prose" vs "aggregate") flags systematically lower-quality rows so
    a model can condition on or exclude the ~47 title-only aggregates.
  - PCA is fit on the L2-normalized vectors (sklearn centers them). BGE space is
    anisotropic; PC1 tends to capture a dominant shared direction -- inspect
    pca_explained_variance.csv and consider dropping pc001 if it hurts a model.
  - Model + corpus provenance is written into the parquet file metadata.

Run: uv run python src/matrix.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.decomposition import PCA

OUT = Path(__file__).resolve().parent.parent / "out"
PCA_VAR_TARGET = 0.90  # keep enough components to explain this much variance
SEED = 42


def write_parquet_with_meta(df: pd.DataFrame, path: Path, meta: dict) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    md = {**(table.schema.metadata or {}), b"provenance": json.dumps(meta).encode()}
    pq.write_table(table.replace_schema_metadata(md), path)


def main() -> None:
    vecs = np.load(OUT / "embeddings.npy")
    meta = pd.read_parquet(OUT / "meta.parquet").reset_index(drop=True)
    manifest = json.loads((OUT / "embeddings_manifest.json").read_text())
    key = meta[["industry_code", "tier"]].copy()

    # --- full-dimension feature matrix ---
    emb_cols = [f"emb_{i:04d}" for i in range(vecs.shape[1])]
    full = pd.concat([key, pd.DataFrame(vecs, columns=emb_cols)], axis=1)
    write_parquet_with_meta(full, OUT / "industry_vectors.parquet", manifest)

    # --- PCA-reduced feature matrix ---
    pca = PCA(random_state=SEED).fit(vecs)
    cum = np.cumsum(pca.explained_variance_ratio_)
    n_comp = int(np.searchsorted(cum, PCA_VAR_TARGET) + 1)
    scores = pca.transform(vecs)[:, :n_comp].astype(np.float32)
    pc_cols = [f"pc{i:03d}" for i in range(1, n_comp + 1)]
    reduced = pd.concat([key, pd.DataFrame(scores, columns=pc_cols)], axis=1)
    write_parquet_with_meta(reduced, OUT / "industry_pca.parquet",
                            {**manifest, "pca_components": n_comp,
                             "pca_var_explained": round(float(cum[n_comp - 1]), 4)})
    reduced.to_csv(OUT / "industry_pca.csv", index=False)

    pd.DataFrame({
        "component": pc_cols + [f"pc{i:03d}" for i in range(n_comp + 1, len(cum) + 1)],
        "explained_variance_ratio": pca.explained_variance_ratio_.round(6),
        "cumulative": cum.round(6),
    }).to_csv(OUT / "pca_explained_variance.csv", index=False)

    # --- authoritative closeness: full-vector cosine (vectors are L2-normalized) ---
    cos = vecs @ vecs.T
    codes = meta["industry_code"].tolist()
    pd.DataFrame(cos, index=codes, columns=codes).to_parquet(OUT / "cosine_similarity.parquet")

    print(f"full matrix:  {full.shape}  -> industry_vectors.parquet")
    print(f"PCA matrix:   {reduced.shape}  ({n_comp} comps, "
          f"{cum[n_comp-1]*100:.1f}% var)  -> industry_pca.parquet/.csv")
    print(f"PC1 alone explains {pca.explained_variance_ratio_[0]*100:.1f}% variance")
    print(f"cosine matrix: {cos.shape}  -> cosine_similarity.parquet")


if __name__ == "__main__":
    main()
