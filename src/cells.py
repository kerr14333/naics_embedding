"""Export the CES State & Area publication cells, decoded and linked to embeddings.

A "publication cell" is a series BLS actually publishes. sm.series lists 22,927 of
them, but that count is inflated by data_type (employment, hours, earnings, ...) and
seasonal-adjustment variants of the same underlying industry x geography cell. This
DEDUPES to the unique (state, area, supersector, industry) cell and decodes every
code to its name, then attaches the industry's NAICS + quality tier so each cell links
straight to its embedding via `industry_code`.

Outputs:
  out/publication_cells.parquet
  out/publication_cells.csv
  docs/publication_cells.csv   (tracked copy for the repo)

Run: uv run python src/cells.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "out"
DOCS = ROOT / "docs"


def read_map(name: str) -> dict:
    df = pd.read_csv(DATA / name, sep="\t", dtype=str).rename(columns=lambda c: c.strip())
    k, v = df.columns[:2]
    return dict(zip(df[k].str.strip(), df[v].str.strip()))


def main() -> None:
    ser = pd.read_csv(DATA / "sm.series", sep="\t", dtype=str).rename(columns=lambda c: c.strip())
    for col in ["state_code", "area_code", "supersector_code", "industry_code"]:
        ser[col] = ser[col].str.strip()

    # Dedupe to the unique geography x industry cell (drop data_type + seasonal).
    cells = (
        ser[["state_code", "area_code", "supersector_code", "industry_code"]]
        .drop_duplicates()
        .sort_values(["state_code", "area_code", "supersector_code", "industry_code"])
        .reset_index(drop=True)
    )

    cells["state_name"] = cells["state_code"].map(read_map("sm.state"))
    cells["area_name"] = cells["area_code"].map(read_map("sm.area"))
    cells["supersector_name"] = cells["supersector_code"].map(read_map("sm.supersector"))
    cells["industry_name"] = cells["industry_code"].map(read_map("sm.industry"))

    # Link to the embedding rows (NAICS + quality tier) by industry_code.
    meta = pd.read_parquet(OUT / "meta.parquet")[["industry_code", "naics", "tier"]]
    cells = cells.merge(meta, on="industry_code", how="left")

    cells = cells[[
        "state_code", "state_name", "area_code", "area_name",
        "supersector_code", "supersector_name",
        "industry_code", "industry_name", "naics", "tier",
    ]]

    DOCS.mkdir(exist_ok=True)
    cells.to_parquet(OUT / "publication_cells.parquet", index=False)
    cells.to_csv(OUT / "publication_cells.csv", index=False)
    cells.to_csv(DOCS / "publication_cells.csv", index=False)

    print(f"raw series: {len(ser):,}   deduped cells: {len(cells):,}")
    print(f"unique: {cells['state_code'].nunique()} states, "
          f"{cells['area_code'].nunique()} areas, "
          f"{cells['industry_code'].nunique()} industries")
    print(f"tier coverage: {cells['tier'].value_counts().to_dict()}")
    print(f"written: out/publication_cells.parquet/.csv + docs/publication_cells.csv")


if __name__ == "__main__":
    main()
