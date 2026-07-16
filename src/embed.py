"""Embed industry sentences with BAAI/bge-large-en-v1.5 (1024-dim).

We compare industry sentences to each other (symmetric similarity + clustering),
so no retrieval query-instruction prefix is used -- every sentence is encoded the
same way. Vectors are L2-normalized, so dot product == cosine similarity.

Outputs:
  out/embeddings.npy  -- float32 [n, 1024], row order matches out/meta.parquet
  out/meta.parquet    -- industry metadata in matching row order

Run: uv run python src/embed.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import sentence_transformers as st
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "out"
MODEL = "BAAI/bge-large-en-v1.5"
MAX_TOKENS = 512  # BGE context window


def token_report(model, sentences) -> int:
    """Count sentences that exceed the model window (would be silently truncated)."""
    tok = model.tokenizer
    lengths = [len(tok.encode(s, add_special_tokens=True)) for s in sentences]
    over = sum(1 for n in lengths if n > MAX_TOKENS)
    print(f"tokens: max {max(lengths)}  over {MAX_TOKENS}: {over}/{len(lengths)}")
    return max(lengths)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    corpus = pd.read_parquet(DATA / "corpus.parquet")
    sentences = corpus["sentence"].tolist()

    print(f"loading {MODEL} ...")
    model = SentenceTransformer(MODEL)
    max_tokens = token_report(model, sentences)

    print(f"encoding {len(corpus)} sentences ...")
    vecs = model.encode(
        sentences,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    np.save(OUT / "embeddings.npy", vecs)
    corpus.to_parquet(OUT / "meta.parquet", index=False)

    # Traceability manifest (review M5): ties these vectors to a model + corpus.
    corpus_hash = hashlib.sha256((DATA / "corpus.parquet").read_bytes()).hexdigest()[:16]
    manifest = {
        "date": date.today().isoformat(),
        "model": MODEL,
        "sentence_transformers": st.__version__,
        "n_rows": len(corpus),
        "dim": int(vecs.shape[1]),
        "max_tokens": int(max_tokens),
        "normalized": True,
        "corpus_sha256_16": corpus_hash,
    }
    (OUT / "embeddings_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nembeddings: {vecs.shape}  -> {OUT / 'embeddings.npy'}")
    print(f"meta:       {len(corpus)} rows -> {OUT / 'meta.parquet'}")
    print(f"manifest:   {manifest}")


if __name__ == "__main__":
    main()
