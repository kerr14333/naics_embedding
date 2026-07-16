"""Fetch raw source files for the NAICS embedding project.

Sources:
  - BLS CES State & Area (survey SM) flat files: industry / supersector / series maps.
  - Census 2022 NAICS Descriptions (rich definitional prose per code).

BLS returns 403 without a descriptive User-Agent, so we set one with a contact
email per BLS data-access policy. Census serves the xlsx without special headers.

Raw files land in data/ (gitignored). Run: uv run python src/fetch.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import requests

DATA = Path(__file__).resolve().parent.parent / "data"

# BLS wants a real UA + contact. Keep the email here so BLS can reach the user.
UA = "naics_embedding research script (cyg5005@gmail.com)"
HEADERS = {"User-Agent": UA}

BLS_BASE = "https://download.bls.gov/pub/time.series/sm/"
BLS_FILES = [
    "sm.industry",     # industry_code -> industry_name
    "sm.supersector",  # supersector_code -> supersector_name
    "sm.series",       # series_id -> state/area/supersector/industry/data_type
    "sm.data_type",    # data_type_code -> data_type_text (context)
]

CENSUS_FILES = {
    "2022_NAICS_Descriptions.xlsx": (
        "https://www.census.gov/naics/2022NAICS/2022_NAICS_Descriptions.xlsx"
    ),
}


def download(url: str, dest: Path) -> None:
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    print(f"  ok  {dest.name:32} {len(resp.content):>10,} bytes")


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {DATA}\n")

    print("BLS CES State & Area (SM):")
    for name in BLS_FILES:
        download(BLS_BASE + name, DATA / name)

    print("\nCensus NAICS:")
    for name, url in CENSUS_FILES.items():
        download(url, DATA / name)

    # Data manifest (review M5): BLS SM files change monthly and are overwritten
    # in place, so stamp date + size + hash to make a given embeddings snapshot
    # traceable to the exact source data it came from.
    names = BLS_FILES + list(CENSUS_FILES)
    manifest = {
        "date": date.today().isoformat(),
        "files": {
            n: {
                "bytes": (DATA / n).stat().st_size,
                "sha256": hashlib.sha256((DATA / n).read_bytes()).hexdigest(),
            }
            for n in names
        },
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("\nDone. Wrote data/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
