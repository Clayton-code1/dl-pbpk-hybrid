"""Step 2 — Fetch canonical SMILES from PubChem for each drug in the Schulz dataset.

Inputs:
    experiments/data/therapeutic_windows/raw/schulz2020_raw_extracted.csv

Outputs:
    experiments/data/therapeutic_windows/smiles_cache.csv   -- persistent cache; re-runs skip already-fetched names
    experiments/data/therapeutic_windows/smiles_results.csv -- matched: drug_name_raw, query_name, cid, smiles, mw, logp, source
    experiments/data/therapeutic_windows/smiles_no_match.csv -- drugs with zero PubChem match; includes reason

Design:
    - Reuses the same PubChem PUG-REST pattern from experiments/data/download_pk_data.py
    - Rate-limited to 1 request per REQUEST_INTERVAL_S (5 requests/s is PubChem's documented limit;
      we use a conservative ~3 req/s = 0.34s gap to be polite and avoid 429s)
    - All results cached to smiles_cache.csv so the script is safely re-runnable
    - Drug name normalisation (multi-stage):
        stage 1: full name as-is
        stage 2: strip parenthetical alternate names like "(Coffein)" from "Caffeine (Coffein)"
        stage 3: strip leading position/descriptor qualifiers like "(4-)" from "Aminobenzoic acid (4-)"
        stage 4: first token only (before any space / delimiter)
        Each stage is only tried if all previous stages failed.
    - No structure guessing: if all stages fail, the drug goes to smiles_no_match.csv with the
      last HTTP status code and the query strings tried.
"""

from __future__ import annotations

import csv
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[2]

SCHULZ_CSV = _HERE / "raw" / "schulz2020_raw_extracted.csv"
CACHE_CSV = _HERE / "smiles_cache.csv"
RESULTS_CSV = _HERE / "smiles_results.csv"
NO_MATCH_CSV = _HERE / "smiles_no_match.csv"

# ---------------------------------------------------------------------------
# PubChem config  (mirrors download_pk_data._try_pubchem)
# ---------------------------------------------------------------------------
PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name"
# PubChem renamed CanonicalSMILES -> ConnectivitySMILES in their property API;
# IsomericSMILES remains available. We request both and use whichever is present.
PUBCHEM_PROPS = "ConnectivitySMILES,IsomericSMILES,MolecularWeight,XLogP,IUPACName"
PUBCHEM_TIMEOUT_S = 10
REQUEST_INTERVAL_S = 0.35   # ~3 req/s — conservative; PubChem asks for <=5 req/s

# ---------------------------------------------------------------------------
# Cache field names
# ---------------------------------------------------------------------------
CACHE_FIELDS = ["drug_name_raw", "query_name", "status", "cid", "smiles", "mw", "logp", "iupac"]
RESULT_FIELDS = ["drug_name_raw", "query_name", "cid", "smiles", "mw", "logp", "iupac", "source"]
NO_MATCH_FIELDS = ["drug_name_raw", "queries_tried", "last_http_status", "note"]

# ---------------------------------------------------------------------------
# Name normalisation helpers
# ---------------------------------------------------------------------------

# Trailing footnote digits (carry-over from PDF extraction — safety net)
_TRAILING_DIGITS = re.compile(r"\s*\d+\s*$")

# Parenthetical content patterns:
#   (Coffein), (Salicylate), etc.  — alternate names to strip
_TRAILING_PAREN = re.compile(r"\s*\([^)]+\)\s*$")
#   Leading position/descriptor: "(4-)", "(2-)", "(5-)", "(N-)"
_LEADING_POSITION = re.compile(r"^\s*\([^)]*[-)\d]\)\s*")

# Drug name aliases not in the raw Schulz text that PubChem knows by different names
# (keyed: schulz_name_fragment -> pubchem_query_name)
MANUAL_ALIASES: dict[str, str] = {
    "paracetamol": "acetaminophen",
    "coffein": "caffeine",
    "epinephrine": "adrenaline",
    "norepinephrine": "noradrenaline",
    "levothyroxine": "thyroxine",
    "acetylsalicylic acid": "aspirin",
    "phenazone": "antipyrine",
    "metamizole": "dipyrone",
    "mesalazine": "mesalamine",
}

# Chemicals that PubChem knows by a synonym — if stage 1-4 fail, try these
SYNONYMS: dict[str, str] = {
    "acetylsalicylic acid (aspirin, ass, asa)": "aspirin",
    "furosemide (frusemide)": "furosemide",
    "salbutamol (albuterol)": "salbutamol",
    "ciclosporin (cyclosporine)": "cyclosporine",
}


def _normalise_stages(raw: str) -> list[str]:
    """Generate candidate query strings for PubChem in order of preference."""
    raw = raw.strip()
    # Safety: strip trailing footnote digits
    raw = _TRAILING_DIGITS.sub("", raw).strip()

    candidates: list[str] = []

    # Stage 1: verbatim (after digit cleanup)
    candidates.append(raw)

    # Stage 2: strip trailing parenthetical "(Alternate Name)"
    s2 = _TRAILING_PAREN.sub("", raw).strip()
    if s2 and s2 != raw:
        candidates.append(s2)

    # Stage 3: additionally strip leading position qualifier "(4-)"
    s3 = _LEADING_POSITION.sub("", s2 or raw).strip()
    if s3 and s3 not in candidates:
        candidates.append(s3)

    # Stage 4: first meaningful token (word before first space or comma or "/")
    s4 = re.split(r"[\s,/]", s3 or raw)[0].strip(" -()")
    if s4 and len(s4) > 2 and s4 not in candidates:
        candidates.append(s4)

    # Stage 5: check manual synonym map
    for fragment, alias in SYNONYMS.items():
        if fragment in raw.lower() and alias not in candidates:
            candidates.append(alias)
    for fragment, alias in MANUAL_ALIASES.items():
        if fragment in raw.lower() and alias not in candidates:
            candidates.append(alias)

    # Deduplicate preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for c in candidates:
        if c.lower() not in seen:
            seen.add(c.lower())
            unique.append(c)
    return unique


# ---------------------------------------------------------------------------
# PubChem REST query  (mirrors download_pk_data._try_pubchem)
# ---------------------------------------------------------------------------

def _pubchem_query(name: str) -> dict[str, Any] | None:
    """Query PubChem PUG REST for a drug name.

    Returns a dict with keys: cid, smiles, mw, logp, iupac, source
    or None if the name is not found (404) or an error occurs.
    """
    url = f"{PUBCHEM_BASE}/{requests.utils.quote(name, safe='')}/property/{PUBCHEM_PROPS}/JSON"
    try:
        r = requests.get(url, timeout=PUBCHEM_TIMEOUT_S)
        if r.status_code == 200:
            props = r.json().get("PropertyTable", {}).get("Properties", [{}])[0]
            # PubChem uses ConnectivitySMILES now; fall back to IsomericSMILES
            smiles = (props.get("ConnectivitySMILES")
                      or props.get("IsomericSMILES")
                      or props.get("CanonicalSMILES")
                      or "")
            return {
                "cid": props.get("CID", ""),
                "smiles": smiles,
                "mw": props.get("MolecularWeight", ""),
                "logp": props.get("XLogP", ""),
                "iupac": props.get("IUPACName", ""),
                "source": "pubchem_rest",
                "http_status": 200,
            }
        return {"cid": "", "smiles": "", "mw": "", "logp": "", "iupac": "",
                "source": "", "http_status": r.status_code}
    except requests.RequestException as exc:
        return {"cid": "", "smiles": "", "mw": "", "logp": "", "iupac": "",
                "source": "", "http_status": f"error:{exc}"}


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def _load_cache(path: Path) -> dict[str, dict]:
    """Load cache keyed by lower-cased drug_name_raw."""
    cache: dict[str, dict] = {}
    if not path.exists():
        return cache
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cache[row["drug_name_raw"].lower()] = row
    return cache


def _append_cache(path: Path, row: dict) -> None:
    """Append one row to the cache CSV, creating headers if file is new."""
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CACHE_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Main fetch loop
# ---------------------------------------------------------------------------

def fetch_all(schulz_csv: Path) -> None:
    if not schulz_csv.exists():
        print(f"ERROR: input CSV not found: {schulz_csv}", file=sys.stderr)
        sys.exit(1)

    # Load unique drug names from Schulz CSV
    with open(schulz_csv, encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    drug_names: list[str] = []
    seen: set[str] = set()
    for r in all_rows:
        n = r["drug_name_raw"].strip()
        if n and n.lower() not in seen:
            seen.add(n.lower())
            drug_names.append(n)

    print(f"Unique drug names to fetch: {len(drug_names)}")

    # Load existing cache
    cache = _load_cache(CACHE_CSV)
    cached_count = sum(1 for n in drug_names if n.lower() in cache)
    print(f"Already cached: {cached_count}  |  To fetch: {len(drug_names) - cached_count}")

    results: list[dict] = []
    no_match: list[dict] = []
    fetched = 0
    skipped_cache = 0

    for idx, drug_name in enumerate(drug_names):
        key = drug_name.lower()

        # Use cache if available
        if key in cache:
            cached_row = cache[key]
            skipped_cache += 1
            if cached_row.get("status") == "match" and cached_row.get("smiles"):
                results.append({
                    "drug_name_raw": drug_name,
                    "query_name": cached_row["query_name"],
                    "cid": cached_row["cid"],
                    "smiles": cached_row["smiles"],
                    "mw": cached_row["mw"],
                    "logp": cached_row["logp"],
                    "iupac": cached_row.get("iupac", ""),
                    "source": "pubchem_rest",
                })
            else:
                no_match.append({
                    "drug_name_raw": drug_name,
                    "queries_tried": cached_row.get("query_name", ""),
                    "last_http_status": cached_row.get("status", ""),
                    "note": "from_cache",
                })
            continue

        # Try candidate query names in order
        candidates = _normalise_stages(drug_name)
        matched = False
        last_status = "never_queried"
        queries_tried: list[str] = []

        for query in candidates:
            queries_tried.append(query)
            time.sleep(REQUEST_INTERVAL_S)
            result = _pubchem_query(query)
            fetched += 1

            if result and result.get("smiles"):
                # Success
                cache_row = {
                    "drug_name_raw": drug_name,
                    "query_name": query,
                    "status": "match",
                    "cid": result["cid"],
                    "smiles": result["smiles"],
                    "mw": result["mw"],
                    "logp": result["logp"],
                    "iupac": result.get("iupac", ""),
                }
                _append_cache(CACHE_CSV, cache_row)
                cache[key] = cache_row
                results.append({
                    "drug_name_raw": drug_name,
                    "query_name": query,
                    "cid": result["cid"],
                    "smiles": result["smiles"],
                    "mw": result["mw"],
                    "logp": result["logp"],
                    "iupac": result.get("iupac", ""),
                    "source": "pubchem_rest",
                })
                matched = True
                break
            else:
                last_status = str(result.get("http_status", "unknown")) if result else "request_failed"
                # Small extra pause on 429 (rate-limit exceeded)
                if result and result.get("http_status") == 429:
                    print(f"  [429] Rate limited on {query!r} — sleeping 5s")
                    time.sleep(5.0)

        if not matched:
            cache_row = {
                "drug_name_raw": drug_name,
                "query_name": "; ".join(queries_tried),
                "status": f"no_match:{last_status}",
                "cid": "", "smiles": "", "mw": "", "logp": "", "iupac": "",
            }
            _append_cache(CACHE_CSV, cache_row)
            cache[key] = cache_row
            no_match.append({
                "drug_name_raw": drug_name,
                "queries_tried": "; ".join(queries_tried),
                "last_http_status": last_status,
                "note": "",
            })

        # Progress report every 50 drugs
        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{len(drug_names)}] fetched={fetched} matched={len(results)} no_match={len(no_match)} cached={skipped_cache}")

    # Write results
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    with open(NO_MATCH_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=NO_MATCH_FIELDS)
        writer.writeheader()
        writer.writerows(no_match)

    print(f"\n=== SMILES fetch complete ===")
    print(f"  Total unique drugs:     {len(drug_names)}")
    print(f"  From cache:             {skipped_cache}")
    print(f"  Newly fetched (API):    {fetched}")
    print(f"  Matched (have SMILES):  {len(results)}")
    print(f"  No match (logged):      {len(no_match)}")
    print(f"\n  Results -> {RESULTS_CSV}")
    print(f"  No-match log -> {NO_MATCH_CSV}")
    print(f"  Cache -> {CACHE_CSV}")

    if no_match:
        print(f"\nSample of first 10 no-match drugs:")
        for r in no_match[:10]:
            print(f"  {r['drug_name_raw'][:50]:<50}  last_status={r['last_http_status']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fetch_all(SCHULZ_CSV)
