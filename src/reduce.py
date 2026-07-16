"""Reduce embeddings to 2D (UMAP) and cluster (HDBSCAN).

UMAP uses cosine metric (matches the normalized-vector similarity used elsewhere).
random_state is set for reproducible layouts (per umap-learn version/platform --
not stable across library upgrades). HDBSCAN runs on a 10-D UMAP projection, not
the raw 1024-D vectors: density estimates degrade badly in high dimensions, so raw
clustering collapsed everything into 2 blobs + noise. Label -1 == noise/unclustered;
membership strength is saved as cluster_prob so weak assignments can be filtered.

Outputs:
  out/coords.parquet          -- industry_code, x, y, cluster, cluster_prob (+ metadata)
  out/plot_supersector.png    -- 2D scatter colored by BLS supersector
  out/plot_cluster.png        -- 2D scatter colored by HDBSCAN cluster

Run: uv run python src/reduce.py
"""

from __future__ import annotations

from pathlib import Path

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

OUT = Path(__file__).resolve().parent.parent / "out"
SEED = 42


def scatter(coords: pd.DataFrame, color_col: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 10))
    groups = list(coords.groupby(color_col))
    # tab20 only has 20 colors; interpolate a larger qualitative ramp so 20+
    # groups don't reuse a hue. Noise cluster (-1) is drawn gray.
    hues = plt.cm.gist_ncar(np.linspace(0, 1, max(len(groups), 1)))
    for (key, g), c in zip(groups, hues):
        gray = str(key) == "-1"
        ax.scatter(g["x"], g["y"], s=22, color=("0.7" if gray else c),
                   label=("noise" if gray else str(key)), alpha=0.8, linewidths=0)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=7, markerscale=1.3, ncol=2, loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path.name}")


def main() -> None:
    vecs = np.load(OUT / "embeddings.npy")
    meta = pd.read_parquet(OUT / "meta.parquet").reset_index(drop=True)

    # 2D projection for plotting.
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=SEED)
    xy = reducer.fit_transform(vecs)

    # Cluster on a low-dim UMAP projection, not raw 1024-dim: HDBSCAN's density
    # estimate degrades in high dimensions (curse of dimensionality), collapsing
    # everything into a couple of blobs + noise. ~10D keeps structure, restores
    # density contrast.
    cluster_space = umap.UMAP(
        n_neighbors=15, min_dist=0.0, n_components=10, metric="cosine", random_state=SEED
    ).fit_transform(vecs)
    # min_samples defaults to min_cluster_size (not 1) for more conservative,
    # stable clusters; probabilities let downstream filter weak members (review M3).
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric="euclidean")
    labels = clusterer.fit_predict(cluster_space.astype(np.float64))
    probs = clusterer.probabilities_

    coords = meta.copy()
    coords["x"], coords["y"] = xy[:, 0], xy[:, 1]
    coords["cluster"] = labels
    coords["cluster_prob"] = probs.round(3)
    coords.to_parquet(OUT / "coords.parquet", index=False)

    n_clusters = len({l for l in labels if l != -1})
    n_noise = int((labels == -1).sum())
    print(f"clusters: {n_clusters}  noise: {n_noise}/{len(labels)}")
    print(f"wrote {OUT / 'coords.parquet'}")

    scatter(coords, "supersector_name", "SM industries by supersector (UMAP)",
            OUT / "plot_supersector.png")
    scatter(coords, "cluster", "SM industries by HDBSCAN cluster (UMAP)",
            OUT / "plot_cluster.png")


if __name__ == "__main__":
    main()
