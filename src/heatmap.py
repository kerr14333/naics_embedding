"""Cluster-ordered cosine-distance heatmap over the 310 SM industries.

Rows/cols are grouped by HDBSCAN cluster (from reduce.py); within each cluster the
order is seriated by hierarchical optimal-leaf-ordering so the on-diagonal blocks
read cleanly. Noise (-1) is pushed to the end. Dark = close (low distance) -> tight
dark blocks on the diagonal are the clusters.

Output: out/heatmap_clustered.png
Run: uv run python src/heatmap.py  (needs out/embeddings.npy + out/coords.parquet)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

OUT = Path(__file__).resolve().parent.parent / "out"


def seriate(dist: np.ndarray) -> np.ndarray:
    """Optimal-leaf-order indices for a small square distance block."""
    n = len(dist)
    if n <= 2:
        return np.arange(n)
    link = linkage(squareform(dist, checks=False), method="average", optimal_ordering=True)
    return leaves_list(link)


def main() -> None:
    vecs = np.load(OUT / "embeddings.npy")
    coords = pd.read_parquet(OUT / "coords.parquet").reset_index(drop=True)
    dist = 1.0 - (vecs @ vecs.T)  # cosine distance
    np.fill_diagonal(dist, 0.0)

    # Order clusters by size (biggest first); noise (-1) last. Seriate within each.
    labels = coords["cluster"].to_numpy()
    real = [c for c in sorted(set(labels), key=lambda c: -(labels == c).sum()) if c != -1]
    order_clusters = real + ([-1] if -1 in labels else [])

    order, blocks = [], []  # blocks: (label, start, size)
    for c in order_clusters:
        idx = np.where(labels == c)[0]
        idx = idx[seriate(dist[np.ix_(idx, idx)])]
        blocks.append((c, len(order), len(idx)))
        order.extend(idx.tolist())

    order = np.array(order)
    D = dist[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(12, 11))
    im = ax.imshow(D, cmap="magma_r", vmin=0, vmax=float(dist.max()), interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine distance (1 - similarity)")

    # Cluster separators + centered labels.
    for _, start, size in blocks:
        for pos in (start, start + size):
            ax.axhline(pos - 0.5, color="white", lw=0.6)
            ax.axvline(pos - 0.5, color="white", lw=0.6)
    ticks = [start + size / 2 for _, start, size in blocks]
    names = [("noise" if c == -1 else str(c)) for c, _, _ in blocks]
    ax.set_xticks(ticks); ax.set_xticklabels(names, fontsize=7, rotation=90)
    ax.set_yticks(ticks); ax.set_yticklabels(names, fontsize=7)
    ax.set_title("SM industries: cosine-distance matrix, ordered by cluster")
    ax.set_xlabel("cluster"); ax.set_ylabel("cluster")
    fig.tight_layout()
    fig.savefig(OUT / "heatmap_clustered.png", dpi=150)
    plt.close(fig)
    print(f"wrote {OUT / 'heatmap_clustered.png'}  ({len(order)} industries, "
          f"{len(blocks)} blocks)")


if __name__ == "__main__":
    main()
