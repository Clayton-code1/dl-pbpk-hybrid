"""Build unit_flagged_review.csv for manual validation of non-mg/L entries.

Input:   raw/schulz2020_raw_extracted.csv
Output:  unit_flagged_review.csv

Columns:
    drug_name_raw       -- Schulz drug name
    field               -- which field is flagged: therapeutic / toxic / comatose
    original_value      -- exact raw string from Schulz table
    detected_unit       -- the non-mg/L unit found (e.g. ng/mL, µg/mL, µmol/L)
    converted_mg_L      -- proposed mg/L value IF a simple ×1000 division applies
                           (ng/mL -> mg/L = ÷1000; µg/mL -> mg/L = ÷1000)
    conversion_note     -- NEEDS-MOLAR-CONVERSION for µmol/L, nmol/L etc. (requires MW);
                           CONVERTED for simple ×1000 cases; UNCLEAR if unit ambiguous
    therapeutic_raw     -- full therapeutic_raw from extraction (for context)

Rules:
    - ng/mL, µg/mL, ng/g : simple ÷1000 to mg/L — show converted value
    - µmol/L, nmol/L, mmol/L : flag as NEEDS-MOLAR-CONVERSION — do NOT convert
    - % : flag as NEEDS-MOLAR-CONVERSION (% of what? — needs clinical clarification)
    - days, weeks, months : these appear in half-life fields, not concentration — skip
    - Any other unit : flag as UNCLEAR
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

_HERE = Path(__file__).resolve().parent
SCHULZ_CSV = _HERE / "raw" / "schulz2020_raw_extracted.csv"
OUT_CSV = _HERE / "unit_flagged_review.csv"

# ---------------------------------------------------------------------------
# Unit detection and conversion
# ---------------------------------------------------------------------------

NON_MGL_PATTERN = re.compile(
    r"\b(ng/[gm][Ll]?|µg/[gm][Ll]?|μg/[gm][Ll]?|nmol/[Ll]|µmol/[Ll]"
    r"|μmol/[Ll]|pmol/[Ll]|mmol/[Ll]|ng·h|%|days?|weeks?|months?)\b",
    re.IGNORECASE,
)

# Units where simple ÷1000 gives mg/L
SIMPLE_DIV1000 = re.compile(r"\b(ng/[gm][Ll]?|µg/[gm][Ll]?|μg/[gm][Ll]?)\b", re.IGNORECASE)
# Units that need molar mass for conversion
NEEDS_MOLAR = re.compile(r"\b(nmol/[Ll]|µmol/[Ll]|μmol/[Ll]|pmol/[Ll]|mmol/[Ll]|%)\b", re.IGNORECASE)
# Half-life units — irrelevant for concentration fields (shouldn't appear, but guard)
TIME_UNITS = re.compile(r"\b(days?|weeks?|months?|hours?|h)\b", re.IGNORECASE)

# Extract leading numeric range from a raw value string
RANGE_RE = re.compile(r"(-?[\d.]+)(?:\s*[-–]\s*(-?[\d.]+))?", re.IGNORECASE)


def _try_convert(raw_value: str, unit: str) -> tuple[str, str]:
    """Return (converted_mg_L, conversion_note) for a flagged value.

    For ng/mL and µg/mL: attempts ÷1000 conversion.
    For µmol/L etc.:      returns ('', 'NEEDS-MOLAR-CONVERSION').
    """
    if NEEDS_MOLAR.search(unit):
        return ("", "NEEDS-MOLAR-CONVERSION")
    if TIME_UNITS.search(unit):
        return ("", "SKIP:TIME-UNIT-IN-CONC-FIELD")
    if SIMPLE_DIV1000.search(unit):
        # Strip the unit text and parse numbers
        stripped = NON_MGL_PATTERN.sub("", raw_value).strip()
        m = RANGE_RE.search(stripped)
        if m:
            try:
                lo = float(m.group(1)) / 1000.0
                hi = float(m.group(2)) / 1000.0 if m.group(2) else lo
            except (TypeError, ValueError):
                return ("", "UNCLEAR:could-not-parse-number")
            if lo == hi:
                converted = f"{lo:.6g}"
            else:
                converted = f"{lo:.6g}-{hi:.6g}"
            return (converted, "CONVERTED:div1000")
        return ("", "UNCLEAR:could-not-parse-number")
    return ("", "UNCLEAR:unrecognised-unit")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build(schulz_csv: Path, out_csv: Path) -> None:
    with open(schulz_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    flagged_rows: list[dict] = []
    for r in rows:
        for field in ("therapeutic", "toxic", "comatose"):
            raw_col = f"{field}_raw"
            raw_val = r.get(raw_col, "").strip()
            if not raw_val:
                continue
            m = NON_MGL_PATTERN.search(raw_val)
            if not m:
                continue
            unit = m.group(1)
            converted, note = _try_convert(raw_val, unit)
            flagged_rows.append({
                "drug_name_raw": r["drug_name_raw"],
                "field": field,
                "original_value": raw_val,
                "detected_unit": unit,
                "converted_mg_L": converted,
                "conversion_note": note,
                "therapeutic_raw": r.get("therapeutic_raw", ""),
            })

    fieldnames = [
        "drug_name_raw", "field", "original_value", "detected_unit",
        "converted_mg_L", "conversion_note", "therapeutic_raw",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flagged_rows)

    print(f"Unit-flagged rows written: {len(flagged_rows)}")
    print(f"Output -> {out_csv}")
    print()
    print("Summary by conversion_note:")
    from collections import Counter
    for note, cnt in Counter(r["conversion_note"] for r in flagged_rows).most_common():
        print(f"  {note:<40} {cnt}")
    print()
    print("All flagged entries:")
    for r in flagged_rows:
        print(f"  {r['drug_name_raw'][:35]:<35} [{r['field']}]  {r['original_value'][:35]:<35}  unit={r['detected_unit']!r:<12}  conv={r['converted_mg_L']!r:<15}  note={r['conversion_note']}")


if __name__ == "__main__":
    build(SCHULZ_CSV, OUT_CSV)
