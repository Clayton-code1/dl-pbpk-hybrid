"""Step 3 — Merge Schulz PK data with PubChem SMILES, produce final dataset.

Inputs:
    raw/schulz2020_raw_extracted.csv   -- Schulz concentrations (Step 1 output)
    smiles_results.csv                 -- PubChem SMILES (Step 2 output)
    unit_flagged_review.csv            -- unit-flagged rows for manual review

Output:
    therapeutic_window_dataset.csv     -- final merged, cleaned dataset
    build_dataset_report.txt           -- drop log + sample

== Column schema ==
    drug_name         -- cleaned drug name (trailing footnote digits stripped)
    smiles            -- canonical SMILES from PubChem
    therapeutic_min_mg_L  -- lower bound of therapeutic ("normal") range, mg/L
    therapeutic_max_mg_L  -- upper bound
    toxic_min_mg_L    -- start of toxic range (from), mg/L; may be blank
    half_life_h       -- half-life lower bound in hours; blank if unavailable
    half_life_h_max   -- half-life upper bound in hours; blank if same as min
    source            -- always "Schulz_2020_CritCare"
    pubchem_cid       -- PubChem CID for the matched structure
    unit_converted    -- "yes" if a ng/mL -> mg/L conversion was applied; "no" otherwise

== Drop logic ==
    1. No therapeutic_min AND no therapeutic_max           -> drop (no usable range)
    2. No SMILES from PubChem                              -> drop (can't use for ML)
    3. therapeutic_raw starts with "see" (cross-ref)       -> drop (already filtered in extract)
    4. unit_flag contains NEEDS-MOLAR-CONVERSION           -> drop (unsafe to include without MW)
    5. unit_flag contains UNCLEAR (div1000 attempted but   -> drop and log
       failed to parse)

== Unit conversion applied ==
    For rows where unit_flagged_review.csv marks CONVERTED:div1000, the
    converted_mg_L value replaces the raw therapeutic_min/max in the final dataset.
    The 'unit_converted' column is set to "yes" for these rows.
    Rows with NEEDS-MOLAR-CONVERSION are dropped (not silently converted).

== Half-life parsing ==
    thalf_raw is a free-text string like "3-11", "appr. 30", "2-4 days", "(6-) 8-31".
    We extract the lower and upper bounds:
      - Values in hours: stored directly
      - Values labelled "days": multiplied by 24
      - Values labelled "weeks", "months": flagged with a conversion note but included
      - Single value: stored as both min and max
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
SCHULZ_CSV = _HERE / "raw" / "schulz2020_raw_extracted.csv"
SMILES_CSV = _HERE / "smiles_results.csv"
UNIT_FLAG_CSV = _HERE / "unit_flagged_review.csv"
OUT_CSV = _HERE / "therapeutic_window_dataset.csv"
REPORT_TXT = _HERE / "build_dataset_report.txt"

# ---------------------------------------------------------------------------
# Drug name cleaning
# ---------------------------------------------------------------------------
_TRAILING_DIGITS = re.compile(r"\s*\d+\s*$")
_MULTI_SPACE = re.compile(r"\s{2,}")

def _clean_name(raw: str) -> str:
    """Strip trailing footnote digits, collapse whitespace, strip punctuation."""
    name = raw.strip()
    name = _TRAILING_DIGITS.sub("", name).strip()
    name = _MULTI_SPACE.sub(" ", name)
    return name


# ---------------------------------------------------------------------------
# Half-life parsing
# ---------------------------------------------------------------------------
# Matches the first numeric range in the t½ string (ignores parentheticals first)
_LEADING_PAREN = re.compile(r"^\s*\([^)]*\)\s*")
_THALF_RANGE = re.compile(r"(-?[\d.]+)\s*[-–]\s*(-?[\d.]+)|(-?[\d.]+)")
_DAYS_UNIT = re.compile(r"\bdays?\b", re.IGNORECASE)
_WEEKS_UNIT = re.compile(r"\bweeks?\b", re.IGNORECASE)
_MONTHS_UNIT = re.compile(r"\bmonths?\b", re.IGNORECASE)


def _parse_thalf(raw: str | None) -> tuple[str, str, str]:
    """Return (thalf_h_min, thalf_h_max, conversion_note).

    conversion_note is empty for plain-hours values, 'converted_from_days' for
    day-converted values, or 'UNCLEAR' if parsing failed.
    """
    if not raw:
        return "", "", ""
    val = _LEADING_PAREN.sub("", raw.strip()).strip()
    if not val:
        return "", "", ""

    # Determine unit multiplier
    note = ""
    mult = 1.0
    if _DAYS_UNIT.search(val):
        mult = 24.0
        note = "converted_from_days"
    elif _WEEKS_UNIT.search(val):
        mult = 24.0 * 7
        note = "converted_from_weeks"
    elif _MONTHS_UNIT.search(val):
        mult = 24.0 * 30
        note = "converted_from_months"

    m = _THALF_RANGE.search(val)
    if not m:
        return "", "", "UNCLEAR"
    try:
        if m.group(1) is not None:
            lo = float(m.group(1)) * mult
            hi = float(m.group(2)) * mult
        else:
            lo = float(m.group(3)) * mult
            hi = lo
        lo_s = f"{lo:.4g}"
        hi_s = f"{hi:.4g}" if hi != lo else ""
        return lo_s, hi_s, note
    except (TypeError, ValueError):
        return "", "", "UNCLEAR"


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

SCHEMA = [
    "drug_name", "smiles", "therapeutic_min_mg_L", "therapeutic_max_mg_L",
    "toxic_min_mg_L", "half_life_h", "half_life_h_max",
    "source", "pubchem_cid", "unit_converted", "data_quality_flag",
]

# Split a converted range string on "-" that is NOT part of scientific notation
# e.g. "5.8e-05-0.000144" -> ["5.8e-05", "0.000144"]
#      "0.001-0.007"       -> ["0.001", "0.007"]
_CONV_RANGE_SPLIT = re.compile(r"(?<![eE])-")


def _split_converted_range(conv: str) -> tuple[str, str]:
    """Split a converted_mg_L string into (min, max), handling sci notation."""
    parts = _CONV_RANGE_SPLIT.split(conv, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return parts[0].strip(), parts[0].strip()


def build(
    schulz_csv: Path,
    smiles_csv: Path,
    unit_flag_csv: Path,
    out_csv: Path,
    report_txt: Path,
) -> None:
    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    with open(schulz_csv, encoding="utf-8") as f:
        schulz_rows = list(csv.DictReader(f))

    smiles_map: dict[str, dict] = {}
    if smiles_csv.exists():
        with open(smiles_csv, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                smiles_map[r["drug_name_raw"].lower()] = r
    else:
        print(f"WARNING: {smiles_csv} not found — no SMILES will be merged.", file=sys.stderr)

    # Build a map: (drug_name_raw.lower(), field) -> converted_mg_L for div1000 conversions
    unit_converted_map: dict[tuple[str, str], str] = {}
    unit_needs_molar: set[str] = set()
    unit_unclear: set[str] = set()
    if unit_flag_csv.exists():
        with open(unit_flag_csv, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                key = (r["drug_name_raw"].lower(), r["field"])
                note = r.get("conversion_note", "")
                if note.startswith("CONVERTED:div1000") and r.get("converted_mg_L"):
                    unit_converted_map[key] = r["converted_mg_L"]
                elif "NEEDS-MOLAR-CONVERSION" in note:
                    unit_needs_molar.add(r["drug_name_raw"].lower())
                elif "UNCLEAR" in note:
                    unit_unclear.add(r["drug_name_raw"].lower())

    # -----------------------------------------------------------------------
    # Process rows
    # -----------------------------------------------------------------------
    out_rows: list[dict] = []
    drop_log: list[dict] = []

    def _drop(r: dict, reason: str) -> None:
        drop_log.append({"drug": r["drug_name_raw"], "reason": reason})

    for r in schulz_rows:
        name_raw = r["drug_name_raw"]
        name_key = name_raw.lower()
        drug_name = _clean_name(name_raw)

        # Rule 4 & 5: unsafe unit flags
        if name_key in unit_needs_molar:
            _drop(r, "unit_flag:NEEDS-MOLAR-CONVERSION (µmol/L or %)")
            continue
        if name_key in unit_unclear:
            _drop(r, "unit_flag:UNCLEAR (could not parse non-mg/L value)")
            continue

        # Rule 1: no therapeutic range
        tmin_raw = r.get("therapeutic_min_mg_L", "")
        tmax_raw = r.get("therapeutic_max_mg_L", "")

        # Apply div1000 unit conversion if available
        unit_converted = "no"
        tmin_conv = unit_converted_map.get((name_key, "therapeutic"))
        if tmin_conv:
            # Split handles scientific notation like "5.8e-05-0.000144"
            tmin_raw, tmax_raw = _split_converted_range(tmin_conv)
            unit_converted = "yes"

        if not tmin_raw and not tmax_raw:
            _drop(r, "no_therapeutic_range")
            continue

        # Rule 2: no SMILES
        smiles_row = smiles_map.get(name_key)
        if not smiles_row or not smiles_row.get("smiles"):
            _drop(r, "no_pubchem_smiles")
            continue

        # Half-life
        thalf_lo, thalf_hi, thalf_note = _parse_thalf(r.get("thalf_raw", ""))

        # Toxic min (also handles ng/mL conversion)
        toxic_raw = r.get("toxic_min_mg_L", "")
        toxic_conv = unit_converted_map.get((name_key, "toxic"))
        if toxic_conv:
            toxic_raw, _ = _split_converted_range(toxic_conv)

        # Data quality flag
        quality_flags: list[str] = []
        try:
            if tmin_raw and float(tmin_raw) < 0:
                # Schulz uses "-X" to mean "up to X mg/L" (upper-bound only, min≈0)
                quality_flags.append("upper_bound_only:therapeutic_min_is_schulz_ceiling")
        except (ValueError, TypeError):
            quality_flags.append("malformed_therapeutic_value")
        if unit_converted == "yes":
            quality_flags.append("unit_converted:ng_per_mL_to_mg_per_L")

        out_rows.append({
            "drug_name": drug_name,
            "smiles": smiles_row["smiles"],
            "therapeutic_min_mg_L": tmin_raw,
            "therapeutic_max_mg_L": tmax_raw,
            "toxic_min_mg_L": toxic_raw,
            "half_life_h": thalf_lo,
            "half_life_h_max": thalf_hi,
            "source": "Schulz_2020_CritCare",
            "pubchem_cid": smiles_row.get("cid", ""),
            "unit_converted": unit_converted,
            "data_quality_flag": "; ".join(quality_flags),
        })

    # -----------------------------------------------------------------------
    # Write output
    # -----------------------------------------------------------------------
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SCHEMA)
        writer.writeheader()
        writer.writerows(out_rows)

    # -----------------------------------------------------------------------
    # Write report
    # -----------------------------------------------------------------------
    from collections import Counter
    drop_counts = Counter(r["reason"] for r in drop_log)

    report_lines: list[str] = []
    report_lines.append("=" * 70)
    report_lines.append("THERAPEUTIC WINDOW DATASET — BUILD REPORT")
    report_lines.append("=" * 70)
    report_lines.append(f"Schulz rows in:        {len(schulz_rows)}")
    report_lines.append(f"Drugs in final dataset:{len(out_rows)}")
    report_lines.append(f"Total dropped:         {len(drop_log)}")
    report_lines.append("")
    report_lines.append("Drop reasons:")
    for reason, cnt in drop_counts.most_common():
        report_lines.append(f"  {reason:<50} {cnt}")
    report_lines.append("")

    # Unit-conversion summary
    unit_yes = sum(1 for r in out_rows if r["unit_converted"] == "yes")
    report_lines.append(f"Rows with unit conversion (ng/mL->mg/L): {unit_yes}")
    report_lines.append(f"Rows excluded for NEEDS-MOLAR-CONVERSION: {len(unit_needs_molar)}")
    report_lines.append(f"Rows excluded for UNCLEAR unit:           {len(unit_unclear)}")
    report_lines.append("")
    report_lines.append("Sample of 15 rows (drug_name, smiles[0:30], t_min, t_max, toxic_min, t½):")
    report_lines.append("-" * 70)
    col_w = [30, 30, 10, 10, 10, 10]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
    report_lines.append(fmt.format("drug_name", "smiles[:30]", "t_min", "t_max", "toxic", "t1/2_h"))
    report_lines.append("-" * 70)
    import random; random.seed(42)
    sample = random.sample(out_rows, min(15, len(out_rows)))
    for row in sample:
        report_lines.append(fmt.format(
            str(row["drug_name"])[:col_w[0]],
            str(row["smiles"])[:col_w[1]],
            str(row["therapeutic_min_mg_L"])[:col_w[2]],
            str(row["therapeutic_max_mg_L"])[:col_w[3]],
            str(row["toxic_min_mg_L"])[:col_w[4]],
            str(row["half_life_h"])[:col_w[5]],
        ))
    report_lines.append("=" * 70)

    report_txt.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\n".join(report_lines))
    print(f"\nFinal dataset -> {out_csv}")
    print(f"Report       -> {report_txt}")


if __name__ == "__main__":
    build(SCHULZ_CSV, SMILES_CSV, UNIT_FLAG_CSV, OUT_CSV, REPORT_TXT)
