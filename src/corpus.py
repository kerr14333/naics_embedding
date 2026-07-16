"""Build one embedding sentence per SM-published industry.

CES SM industry_code layout: [2-digit supersector][6-digit NAICS, right-zero-padded].
NAICS = industry_code[2:8], matched to the longest *structurally valid* Census code
whose trimmed tail is all zeros. Custom CES aggregates (Total Nonfarm, Goods
Producing, supersectors, Durable/Non-Durable Goods) have no single NAICS -> title-only.

Prose recovery (fixes review H1): 154 Census 4-digit codes carry a NaN Description
(the prose lives at the 6-digit level), and 522 rows are "See industry description
for XXXXXX" pointers. So a structural match is NOT enough -- resolve_prose() follows
pointers and, for prose-less levels, concatenates the descriptions of the 6-digit
children. Cross-reference tails ("...are classified in Industry X", review M1/H3)
are truncated because they inject other industries' vocabulary and blow the token
budget.

Government (supersector 90) uses pseudo-NAICS (91x/92x/93x); real-NAICS matches
there are false positives (review H2), so we skip Census matching for it entirely.

Design decision (B): supersector is METADATA only (used later for plot color),
never embedded. The industry's OWN title stays in the sentence.

Output: data/corpus.parquet  (with columns: ..., description, tier)
Run: uv run python src/corpus.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"

POINTER = re.compile(r"See industry description for\s+(\d+)", re.IGNORECASE)
# Cut the cross-reference / exclusion tail; keep only the defining prose.
CROSSREF = re.compile(r"\s*Cross-References\.?.*$", re.IGNORECASE | re.DOTALL)
PROSE_BUDGET = 700  # chars; keeps every sentence under BGE's 512-token window (H3)


def compact(text: str, budget: int = PROSE_BUDGET) -> str:
    """First whole sentences up to a char budget -- the defining identity, no tail."""
    if len(text) <= budget:
        return text
    out = ""
    for piece in re.split(r"(?<=\.)\s+", text):
        if out and len(out) + len(piece) + 1 > budget:
            break
        out = f"{out} {piece}".strip()
    return out or text[:budget]


def load_bls():
    ind = pd.read_csv(DATA / "sm.industry", sep="\t", dtype=str).rename(columns=str.strip)
    sup = pd.read_csv(DATA / "sm.supersector", sep="\t", dtype=str).rename(columns=str.strip)
    ser = pd.read_csv(DATA / "sm.series", sep="\t", dtype=str).rename(columns=lambda c: c.strip())
    ind["industry_code"] = ind["industry_code"].str.strip()
    ind["industry_name"] = ind["industry_name"].str.strip()
    sup["supersector_name"] = sup["supersector_name"].str.strip()
    published = set(ser["industry_code"].str.strip().unique())  # SM-only scope
    return ind, sup, published


class Naics:
    """Census NAICS structure + prose, with pointer/child recovery."""

    def __init__(self, desc: pd.DataFrame):
        desc = desc.copy()
        desc["Code"] = desc["Code"].astype(str)
        self.codes = set(desc["Code"])  # all structurally valid codes
        self.raw = dict(zip(desc["Code"], desc["Description"]))
        self.children6 = {}  # prefix -> list of 6-digit descendant codes
        for c in self.codes:
            if len(c) == 6:
                self.children6.setdefault(c[:4], []).append(c)

    def match(self, industry_code: str) -> str | None:
        """Structural NAICS code for a CES industry_code, or None for aggregates."""
        n6 = industry_code[2:8]
        if n6 == "000000":
            return None
        for length in (6, 5, 4, 3, 2):
            cand = n6[:length]
            if (length == 6 or n6[length:] == "0" * (6 - length)) and cand in self.codes:
                return cand
        return None

    def _clean(self, text) -> str:
        if not isinstance(text, str):
            return ""
        text = CROSSREF.sub("", text)
        # Drop sector-summary header noise seen in 2-digit rows.
        text = re.sub(r"\bThe Sector as a Whole\b", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text)  # collapse newlines / runs of whitespace
        return text.strip()

    def prose(self, code: str, _seen=None) -> str:
        """Definitional prose for a code: follow pointers, recover from children."""
        _seen = _seen or set()
        if code in _seen:
            return ""
        _seen.add(code)
        raw = self.raw.get(code)
        if isinstance(raw, str):
            ptr = POINTER.search(raw)
            if ptr:
                return self.prose(ptr.group(1), _seen)
            cleaned = self._clean(raw)
            if cleaned:
                return cleaned
        # prose-less (NaN 4-digit): concatenate 6-digit children's prose.
        parts, seen_txt = [], set()
        for child in sorted(self.children6.get(code, [])):
            p = self.prose(child, _seen)
            if p and p not in seen_txt:
                seen_txt.add(p)
                parts.append(p)
        return " ".join(parts)


def main() -> None:
    ind, sup, published = load_bls()
    desc = pd.read_excel(DATA / "2022_NAICS_Descriptions.xlsx", dtype=str)
    naics = Naics(desc)
    sup_name = dict(zip(sup["supersector_code"], sup["supersector_name"]))

    ind = ind[ind["industry_code"].isin(published)].copy()  # SM-published only
    ind["supersector_code"] = ind["industry_code"].str[:2]
    ind["supersector_name"] = ind["supersector_code"].map(sup_name)

    def resolve(code: str):
        if code[:2] == "90":  # government pseudo-NAICS -> title only (H2)
            return None, ""
        n = naics.match(code)
        return n, (compact(naics.prose(n)) if n else "")

    ind["naics"], ind["description"] = zip(*ind["industry_code"].map(resolve))

    def tier(row) -> str:
        if row["description"]:
            return "prose"
        return "aggregate" if pd.isna(row["naics"]) else "title-only"

    ind["tier"] = ind.apply(tier, axis=1)
    ind["sentence"] = ind.apply(
        lambda r: f"{r['industry_name']}. {r['description']}" if r["description"] else r["industry_name"],
        axis=1,
    )

    out = ind[[
        "industry_code", "industry_name", "supersector_code", "supersector_name",
        "naics", "description", "tier", "sentence",
    ]].reset_index(drop=True)
    out.to_parquet(DATA / "corpus.parquet", index=False)

    counts = out["tier"].value_counts().to_dict()
    print(f"corpus rows: {len(out)}  (SM-published industries)")
    print(f"tiers: {counts}")
    print(f"max sentence chars: {out['sentence'].str.len().max()}")
    print(f"written: {DATA / 'corpus.parquet'}")


if __name__ == "__main__":
    main()
