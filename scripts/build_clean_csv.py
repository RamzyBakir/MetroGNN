"""Rebuild the clean DBLP CSV from the full raw ``V18-1-20-25`` CSV.

The pipeline trains on a simplified, well-formed CSV with columns
``title, venue, authors, year, n_citation, keywords`` where ``authors`` is a
serialized list of author *names*. The full raw CSV stores authors as a list of
``{id, name, org}`` dicts and, crucially, has broken quote-escaping inside the
``abstract`` field that defeats a normal CSV parser. Since each record is a
single physical line with a fixed field order, the needed fields are recovered
with positional anchors instead of a full CSV parse:

    id, title, "abstract...", "['keywords']", YEAR, "[{authors}]",
    "[references]", "[urls]", N_CITATION, venue, doc_type

Usage:
    python scripts/build_clean_csv.py
    python scripts/build_clean_csv.py --full train/raw.csv --out train/clean.csv
"""

import argparse
import ast
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

COLUMNS = ["title", "venue", "authors", "year", "n_citation", "keywords"]
MIN_YEAR = 2020
MAX_YEAR = 2025

TITLE_RE = re.compile(r'^[^,]*,(.*?),\"')                 # id, title, "abstract...
KEYWORDS_RE = re.compile(r'\"(\[(?:[^\[\]]*)\])\",(20[12][0-9]),')  # "['kw']",YEAR,
AUTHORS_RE = re.compile(r',(20[12][0-9]),\"(\[\{.*?\}\])\",')        # ,YEAR,"[{...}]",


def parse_list(value):
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError, TypeError):
        return None
    return parsed if isinstance(parsed, list) else None


def extract_record(line):
    line = line.rstrip("\n")
    ym = AUTHORS_RE.search(line)
    if not ym:
        return None
    year = ym.group(1)

    tm = TITLE_RE.match(line)
    title = tm.group(1).strip() if tm else ""
    if not title:
        return None

    authors_raw = parse_list(ym.group(2))
    if authors_raw is None:
        return None
    authors = [
        str(a.get("name", "")).strip()
        for a in authors_raw
        if isinstance(a, dict) and str(a.get("name", "")).strip()
    ]

    keywords = []
    km = KEYWORDS_RE.search(line)
    if km:
        kw = parse_list(km.group(1))
        if kw:
            keywords = [str(k).strip() for k in kw if str(k).strip()]

    # tail: ...,N_CITATION,venue,doc_type  (venues contain no commas).
    # Many rows carry a run of trailing empty fields, so strip them first.
    tail = line.rstrip(",")
    prefix, venue, _doc_type = tail.rsplit(",", 2)
    n_citation = prefix.rsplit(",", 1)[1].strip()

    return {
        "title": title,
        "venue": venue.strip(),
        "authors": str(authors),
        "year": int(year),
        "n_citation": n_citation,
        "keywords": str(keywords),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        default=str(PROJECT_ROOT / "train" / "DBLP-Citation-network-V18-1-20-25.csv"),
        help="Path to the full raw V18-1-20-25 CSV",
    )
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "train" / "DBLP-Citation-network-V18-1-20-25_clean.csv"),
        help="Output path for the clean CSV",
    )
    args = parser.parse_args()

    records = []
    seen = set()
    skipped = 0
    with open(args.full, encoding="utf-8", errors="replace") as fh:
        fh.readline()  # header
        for line in fh:
            rec = extract_record(line)
            if rec is None or not (MIN_YEAR <= rec["year"] <= MAX_YEAR):
                skipped += 1
                continue
            key = (rec["title"].casefold(), rec["year"])
            if key in seen:
                continue
            seen.add(key)
            records.append(rec)

    if not records:
        raise SystemExit("No records extracted; check the raw CSV path.")

    clean = pd.DataFrame(records, columns=COLUMNS)
    clean.to_csv(args.out, index=False)
    print(f"Extracted {len(clean)} clean rows ({skipped} unparseable lines skipped)")
    print(f"Year range: {clean.year.min()}-{clean.year.max()}")
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
