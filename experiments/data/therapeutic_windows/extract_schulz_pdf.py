"""Step 1 — Extract Schulz 2020 supplementary table from PDF.

Input:  experiments/data/therapeutic_windows/raw/schulz2020_supplementary_table.pdf
Output: experiments/data/therapeutic_windows/raw/schulz2020_raw_extracted.csv

Source:
  Schulz et al. (2020), Critical Care, doi:10.1186/s13054-020-02915-5
  Additional File 1: therapeutic/toxic blood concentrations of >1,100 drugs

== Table layout (pages 1-84; pages 85-208 are footnotes) ==

pdfplumber find_tables() returns rows with 17 cells per row due to merged headers.
Active column indices:
    0  = Substance (drug name)
    3  = Therapeutic ("normal") concentration  [mg/L]
    6  = Toxic (from)                          [mg/L]
    9  = Comatose-fatal (from)                 [mg/L]
    12 = t½ (h)
    15 = References

== Footnote handling ==

Footnote reference numbers appear as superscript text (font size 6.5pt) directly
appended to concentration values.  Normal data text is 10pt.  We extract each cell
by reading individual characters from pdfplumber and discarding any character with
font size <= SUPERSCRIPT_THRESHOLD (7.5pt), which cleanly separates "308" in "3.9308"
from the actual value "3.9".

== Parenthetical notation ==

Schulz uses "(X-) Y-Z (-W)" to denote extended ranges:
    (X-)  = extended lower boundary
    Y-Z   = MAIN therapeutic range
    (-W)  = extended upper boundary

We strip any leading parenthetical before range parsing, so "(5-) 8-15 (-20)"
yields min=8, max=15 (the main range).

== Unit flags ==

Non-mg/L units (ng/mL, µg/mL, µmol/L, etc.) are flagged in 'unit_flag'; values are
preserved raw and NOT silently converted.  Unit conversions happen in the merge step.
"""

from __future__ import annotations

import csv
import re
import sys
from itertools import groupby
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[2]
_PDF = _HERE / "raw" / "schulz2020_supplementary_table.pdf"
_OUT_CSV = _HERE / "raw" / "schulz2020_raw_extracted.csv"

# Pages 1-84 contain drug data; 85-208 are footnote/reference pages (0-indexed 0-83)
DATA_PAGE_RANGE = range(0, 84)

# Column indices in the 17-column pdfplumber table
COL_NAME = 0
COL_THERAPEUTIC = 3
COL_TOXIC = 6
COL_COMATOSE = 9
COL_THALF = 12
COL_REFS = 15

# Characters with font size at or below this threshold are superscript footnotes
SUPERSCRIPT_THRESHOLD = 7.5  # normal text = 10pt; superscripts = 6.5pt

# ---------------------------------------------------------------------------
# Header row detection
# ---------------------------------------------------------------------------
HEADER_KEYWORDS = {
    "substance", "blood-plasma", "concentration", "therapeutic",
    "toxic", "comatose", "normal", "references",
}


def _is_header_row(cells: list[str]) -> bool:
    text = " ".join(c.lower() for c in cells if c).strip()
    return any(kw in text for kw in HEADER_KEYWORDS) or not text


# ---------------------------------------------------------------------------
# Non-mg/L unit detection
# ---------------------------------------------------------------------------
NON_MGL_PATTERN = re.compile(
    r"\b(ng/[gm][Ll]?|µg/[gm][Ll]?|μg/[gm][Ll]?|nmol/[Ll]|µmol/[Ll]"
    r"|μmol/[Ll]|pmol/[Ll]|mmol/[Ll]|ng·h|%|days?|weeks?|months?)\b",
    re.IGNORECASE,
)

APPROX_PREFIX = re.compile(r"^(appr\.?\s*|approximately\s*)", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Cross-reference detection
# ---------------------------------------------------------------------------
SEE_RE = re.compile(r"^\s*see\b", re.IGNORECASE)


def _is_cross_ref_name(name: str) -> bool:
    return bool(name and SEE_RE.match(name))


def _is_cross_ref_value(val: str) -> bool:
    """True when the concentration cell itself starts with 'see X'."""
    return bool(val and SEE_RE.match(val.strip()))


# ---------------------------------------------------------------------------
# Cell text extraction — font-size filtered
# ---------------------------------------------------------------------------

def _extract_cell_text(page_chars: list, bbox: tuple, threshold: float = SUPERSCRIPT_THRESHOLD) -> str:
    """Return cell text with superscript footnote characters removed.

    Filters by font size: characters with size <= threshold are discarded.
    Remaining characters are sorted by (row, x-position) and joined,
    with a space inserted wherever there is a horizontal gap > 2pt.
    """
    x0, top, x1, bottom = bbox
    # 1pt tolerance on all sides
    cell_chars = [
        c for c in page_chars
        if c["x0"] >= x0 - 1 and c["x1"] <= x1 + 1
        and c["top"] >= top - 1 and c["bottom"] <= bottom + 1
        and c.get("size", 10.0) > threshold
        and c["text"].strip()  # skip spaces stored as chars
    ]
    if not cell_chars:
        return ""

    # Sort by approximate row (round top to 1pt) then x-position
    cell_chars.sort(key=lambda c: (round(c["top"], 0), c["x0"]))

    # Group into visual lines by y-position
    line_texts = []
    for _y, group in groupby(cell_chars, key=lambda c: round(c["top"], 0)):
        row_chars = sorted(group, key=lambda c: c["x0"])
        parts: list[str] = []
        prev_x1: float | None = None
        for c in row_chars:
            if prev_x1 is not None and c["x0"] - prev_x1 > 2.0:
                parts.append(" ")
            parts.append(c["text"])
            prev_x1 = c["x0"] + c.get("width", c["size"] * 0.5)
        line_texts.append("".join(parts))

    return " ".join(line_texts).strip()


# ---------------------------------------------------------------------------
# Numeric range parsing
# ---------------------------------------------------------------------------

# Matches a leading parenthetical like "(5-)", "(2-)", "(5-15)" at start of string
LEADING_PAREN = re.compile(r"^\s*\([^)]*\)\s*")

# Matches a blood-concentration numeric range or single value
RANGE_RE = re.compile(
    r"(?:appr\.?\s*|[<>≥≤]\s*)?(-?[\d.]+)\s*[-–]\s*(-?[\d.]+)"
    r"|(?:appr\.?\s*|[<>≥≤]\s*)?(-?[\d.]+)",
    re.IGNORECASE,
)


def _parse_range(raw: str | None) -> tuple[str | None, str | None]:
    """Return (min_str, max_str) from a raw Schulz concentration string.

    Returns (None, None) for empty, non-numeric, or cross-reference values.
    For a single value (e.g. ">=4.9") min == max.

    Handling of parenthetical notation:
      "(5-) 8-15 (-20)" -> strips "(5-)" first -> finds "8-15" as main range
    """
    if not raw:
        return None, None
    val = raw.strip()
    if not val or val in {"-", "—", "–"}:
        return None, None
    if _is_cross_ref_value(val):
        return None, None

    # Strip approximate prefix before parenthetical check
    val = APPROX_PREFIX.sub("", val).strip()

    # Remove leading parenthetical (extended lower bound indicator)
    val = LEADING_PAREN.sub("", val).strip()

    if not val:
        return None, None

    m = RANGE_RE.search(val)
    if not m:
        return None, None

    if m.group(1) is not None:  # two-number range matched
        return m.group(1), m.group(2)
    else:  # single value matched
        single = m.group(3)
        return single, single


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract(pdf_path: Path, out_csv: Path) -> None:
    try:
        import pdfplumber
    except ImportError:
        print("ERROR: pdfplumber not installed.  Run: pip install pdfplumber", file=sys.stderr)
        sys.exit(1)

    rows_out: list[dict] = []
    skipped_header = 0
    skipped_cross_ref = 0
    skipped_empty = 0
    pages_with_table = 0
    pages_without_table = 0

    TABLE_SETTINGS = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "snap_tolerance": 3,
        "join_tolerance": 3,
    }

    with pdfplumber.open(str(pdf_path)) as pdf:
        total_pages = len(pdf.pages)
        print(f"PDF: {total_pages} total pages.  Extracting drug-table pages 1-{DATA_PAGE_RANGE.stop}.")

        for pg_idx in DATA_PAGE_RANGE:
            if pg_idx >= total_pages:
                break
            page = pdf.pages[pg_idx]
            page_chars = page.chars  # pre-load chars once per page

            tables = page.find_tables(table_settings=TABLE_SETTINGS)
            if not tables:
                pages_without_table += 1
                continue

            pages_with_table += 1
            table = tables[0]

            for row in table.rows:
                cells = row.cells  # list of (x0,top,x1,bottom) or None per column

                # Build text list using font-size-filtered char extraction
                cell_texts: list[str] = []
                for bbox in cells:
                    if bbox is None:
                        cell_texts.append("")
                    else:
                        cell_texts.append(_extract_cell_text(page_chars, bbox))

                if _is_header_row(cell_texts):
                    skipped_header += 1
                    continue

                # Pad to at least COL_REFS + 1 entries
                while len(cell_texts) <= COL_REFS:
                    cell_texts.append("")

                raw_name = cell_texts[COL_NAME].strip()
                if not raw_name:
                    skipped_empty += 1
                    continue
                if _is_cross_ref_name(raw_name):
                    skipped_cross_ref += 1
                    continue

                raw_therapeutic = cell_texts[COL_THERAPEUTIC]
                raw_toxic = cell_texts[COL_TOXIC]
                raw_comatose = cell_texts[COL_COMATOSE]
                raw_thalf = cell_texts[COL_THALF]
                raw_refs = cell_texts[COL_REFS]

                # Cross-ref in therapeutic cell (e.g. "see Paracetamol")
                if _is_cross_ref_value(raw_therapeutic):
                    skipped_cross_ref += 1
                    continue

                # Unit flagging — before any numeric extraction
                unit_flag_parts: list[str] = []
                for field_name, field_val in [
                    ("therapeutic", raw_therapeutic),
                    ("toxic", raw_toxic),
                    ("comatose", raw_comatose),
                ]:
                    m = NON_MGL_PATTERN.search(field_val)
                    if m:
                        unit = m.group(1)
                        # µmol/L and nmol/L require molar-mass conversion, not simple division
                        if re.search(r"mol/[Ll]", unit, re.IGNORECASE):
                            unit_flag_parts.append(f"NEEDS-MOLAR-CONVERSION:{field_name}:{field_val!r}")
                        else:
                            unit_flag_parts.append(f"NON-MGL:{field_name}:{unit!r}:{field_val!r}")

                tmin, tmax = _parse_range(raw_therapeutic)
                toxic_min, toxic_max = _parse_range(raw_toxic)
                comatose_min, _ = _parse_range(raw_comatose)

                rows_out.append({
                    "drug_name_raw": raw_name,
                    "therapeutic_raw": raw_therapeutic,
                    "therapeutic_min_mg_L": tmin,
                    "therapeutic_max_mg_L": tmax,
                    "toxic_raw": raw_toxic,
                    "toxic_min_mg_L": toxic_min,
                    "toxic_max_mg_L": toxic_max,
                    "comatose_raw": raw_comatose,
                    "comatose_min_mg_L": comatose_min,
                    "thalf_raw": raw_thalf,
                    "references": raw_refs,
                    "unit_flag": "; ".join(unit_flag_parts),
                    "source": "Schulz_2020_CritCare",
                    "page": pg_idx + 1,
                })

    print(f"\nExtraction summary:")
    print(f"  Pages with table:        {pages_with_table}")
    print(f"  Pages without table:     {pages_without_table}")
    print(f"  Header rows skipped:     {skipped_header}")
    print(f"  Cross-ref rows skipped:  {skipped_cross_ref}")
    print(f"  Empty-name rows skipped: {skipped_empty}")
    print(f"  Drug rows extracted:     {len(rows_out)}")

    if not rows_out:
        print("ERROR: no rows extracted.", file=sys.stderr)
        sys.exit(1)

    fieldnames = [
        "drug_name_raw", "therapeutic_raw", "therapeutic_min_mg_L", "therapeutic_max_mg_L",
        "toxic_raw", "toxic_min_mg_L", "toxic_max_mg_L",
        "comatose_raw", "comatose_min_mg_L",
        "thalf_raw", "references", "unit_flag", "source", "page",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\nRaw extracted CSV -> {out_csv}")
    _print_sample(rows_out, n=10)


# ---------------------------------------------------------------------------
# Sample printing
# ---------------------------------------------------------------------------

def _print_sample(rows: list[dict], n: int = 10) -> None:
    print(f"\nSample of first {n} rows:")
    header = ["drug_name_raw", "therapeutic_min", "therapeutic_max", "toxic_min", "thalf_raw", "unit_flag"]
    col_w = [35, 16, 16, 12, 16, 30]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
    print(fmt.format(*header))
    print("-" * (sum(col_w) + 2 * len(col_w)))
    for row in rows[:n]:
        print(fmt.format(
            str(row["drug_name_raw"])[:col_w[0]],
            str(row["therapeutic_min_mg_L"] or "")[:col_w[1]],
            str(row["therapeutic_max_mg_L"] or "")[:col_w[2]],
            str(row["toxic_min_mg_L"] or "")[:col_w[3]],
            str(row["thalf_raw"])[:col_w[4]],
            str(row["unit_flag"])[:col_w[5]],
        ))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not _PDF.exists():
        print(
            f"ERROR: PDF not found at {_PDF}\n"
            "Please download it and place it at that path before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)
    extract(_PDF, _OUT_CSV)
