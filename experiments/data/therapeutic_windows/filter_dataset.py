"""Step 4 — Filter and flag therapeutic_window_dataset.csv.

Reads:  experiments/data/therapeutic_windows/therapeutic_window_dataset.csv
Writes: experiments/data/therapeutic_windows/therapeutic_window_dataset_filtered.csv

Two operations:
  1. Keep only rows where BOTH therapeutic_min_mg_L AND therapeutic_max_mg_L are
     present and strictly positive.  This drops the 66 upper-bound-only rows
     (Schulz '-X' notation) and any toxic-only entries.

  2. Add column 'likely_therapeutic_agent' = True/False.
     Flagged False for substances in three categories:
       - Industrial solvents / non-drug chemicals
       - Toxic metals and environmental elements with no approved therapeutic use
       - Illicit / NPS / psychedelics with no current regulatory approval
       - Topical sunscreen agents (not systemic therapeutics)
       - Diagnostic / contrast agents
     Legitimate therapeutics that superficially look 'suspicious' are kept True:
     Fentanyl, Ketamine, Cocaine (topical ENT use), Amphetamine (ADHD),
     Dronabinol (FDA antiemetic), Gold (antirheumatic), Iron/Magnesium/Zinc/
     Selenium (supplement therapy), Nicotine (NRT), etc.

Run from project root:
    python -m experiments.data.therapeutic_windows.filter_dataset
"""

from __future__ import annotations

import csv
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_IN_CSV  = _HERE / "therapeutic_window_dataset.csv"
_OUT_CSV = _HERE / "therapeutic_window_dataset_filtered.csv"

# ---------------------------------------------------------------------------
# Non-therapeutic entries — exact name prefixes / substrings
# (matched case-insensitively against drug_name)
# ---------------------------------------------------------------------------

# Each tuple: (match_string, reason_category)
# Matched if the drug_name starts with or equals match_string (lower-cased)
NON_THERAPEUTIC: list[tuple[str, str]] = [
    # --- Industrial solvents / non-drug chemicals ---
    ("acetone",                         "industrial_solvent"),
    ("ammonia",                         "industrial_chemical"),
    ("2,2,2-trichloroethanol",          "industrial_chemical"),   # metabolite, not a drug
    ("chloroform",                      "discontinued_anesthetic_or_solvent"),
    ("cyclopropane",                    "discontinued_anesthetic"),
    ("n-hexane",                        "industrial_solvent"),
    ("propylene glycol",                "excipient_not_therapeutic"),

    # --- Toxic metals / environmental elements (no approved therapeutic use) ---
    ("aluminium",                       "toxic_metal"),
    ("arsenic",                         "toxic_metal"),
    ("boron",                           "element_no_therapeutic_use"),
    ("cadmium",                         "toxic_metal"),
    ("chromium",                        "toxic_metal"),
    ("cobalt",                          "toxic_metal"),           # no therapeutic use as element
    ("manganese",                       "toxic_metal"),
    ("mercury",                         "toxic_metal"),
    ("tin",                             "toxic_metal"),
    ("uranium",                         "toxic_metal"),

    # --- NPS / illicit drugs with no current regulatory approval as therapeutics ---
    ("5-(2-aminopropyl)benzofuran",     "nps_designer_drug"),     # 5-APB
    ("cathinone",                       "nps_precursor_no_approval"),
    ("cathine",                         "controlled_no_fda_approval"),
    ("lysergide",                       "psychedelic_schedule_i"),  # LSD
    ("mescaline",                       "psychedelic_schedule_i"),
    ("methaqualone",                    "schedule_i_depressant"),   # Quaalude
    ("mitragynine",                     "nps_no_approval"),          # Kratom
    ("n,n-dimethyltryptamine",          "psychedelic_schedule_i"),  # DMT
    ("n-benzylpiperazine",              "nps_designer_drug"),        # BZP
    ("phencyclidine",                   "schedule_ii_no_therapeutic_use"),  # PCP
    ("psilocin",                        "psychedelic_schedule_i"),
    ("salvinorin a",                    "psychedelic_no_approval"),

    # --- Topical sunscreen agents (not systemic therapeutic drugs) ---
    ("avobenzone",                      "sunscreen_topical_not_systemic"),
    ("ecamsule",                        "sunscreen_topical_not_systemic"),   # Mexoryl SX
    ("homosalate",                      "sunscreen_topical_not_systemic"),
    ("octinoxate",                      "sunscreen_topical_not_systemic"),
    ("octisalate",                      "sunscreen_topical_not_systemic"),
    ("octocrylene",                     "sunscreen_topical_not_systemic"),
    ("oxybenzone",                      "sunscreen_topical_not_systemic"),

    # --- Diagnostic / imaging agents (not therapeutic) ---
    ("adipiodone",                      "diagnostic_contrast_agent"),
]

# Build a lookup: lower-cased prefix → reason
_NON_THERAPEUTIC_LOOKUP = [(s.lower(), reason) for s, reason in NON_THERAPEUTIC]


def _classify(drug_name: str) -> tuple[bool, str]:
    """Return (likely_therapeutic_agent, reason_if_false)."""
    name_lo = drug_name.lower()
    for prefix, reason in _NON_THERAPEUTIC_LOOKUP:
        if not name_lo.startswith(prefix):
            continue
        # Require that the match covers the full name, or is followed by a
        # non-alphabetic character (space, comma, parenthesis, digit, end).
        # Prevents "tin" from matching "tinidazol", "boron" from matching
        # "borate", etc.
        remainder = name_lo[len(prefix):]
        if remainder == "" or not remainder[0].isalpha():
            return False, reason
    return True, ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    with open(_IN_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    fieldnames_in = list(rows[0].keys()) if rows else []

    def safe_float(s: str) -> float | None:
        try:
            return float(s)
        except (ValueError, TypeError):
            return None

    # Step 1: positive-min AND positive-max filter
    filtered: list[dict] = []
    n_dropped_no_range = 0
    n_dropped_upper_bound = 0

    for r in rows:
        tmin = safe_float(r.get("therapeutic_min_mg_L", ""))
        tmax = safe_float(r.get("therapeutic_max_mg_L", ""))
        if tmin is None or tmax is None:
            n_dropped_no_range += 1
            continue
        if tmin <= 0 or tmax <= 0:
            n_dropped_upper_bound += 1
            continue
        filtered.append(r)

    # Step 2: add likely_therapeutic_agent column
    n_flagged_false = 0
    out_rows: list[dict] = []
    flagged_names: list[tuple[str, str]] = []

    for r in filtered:
        is_therapeutic, reason = _classify(r["drug_name"])
        if not is_therapeutic:
            n_flagged_false += 1
            flagged_names.append((r["drug_name"], reason))
        r_out = dict(r)
        r_out["likely_therapeutic_agent"] = str(is_therapeutic)
        r_out["non_therapeutic_reason"] = reason
        out_rows.append(r_out)

    n_clean = len(out_rows) - n_flagged_false

    # Write output
    out_fields = fieldnames_in + ["likely_therapeutic_agent", "non_therapeutic_reason"]
    with open(_OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(out_rows)

    # --- Report ---
    total_in = len(rows)
    print("=" * 70)
    print("FILTER REPORT")
    print("=" * 70)
    print(f"Input rows (therapeutic_window_dataset.csv):  {total_in}")
    print(f"Dropped — missing min or max:                 {n_dropped_no_range}")
    print(f"Dropped — upper-bound-only (min <= 0):        {n_dropped_upper_bound}")
    print(f"Rows passing filter:                          {len(out_rows)}")
    print(f"  Flagged likely_therapeutic_agent=False:     {n_flagged_false}")
    print(f"  Clean therapeutic drugs (True):             {n_clean}")
    print()
    print("Flagged non-therapeutic entries:")
    for name, reason in sorted(flagged_names, key=lambda x: x[0].lower()):
        print(f"  FALSE  {name[:50]:<50}  [{reason}]")

    # 15-row sample from clean therapeutics
    clean_rows = [r for r in out_rows if r["likely_therapeutic_agent"] == "True"]
    # Sort by drug name for a deterministic, alphabetically spread sample
    clean_rows_sorted = sorted(clean_rows, key=lambda r: r["drug_name"].lower())
    step = max(1, len(clean_rows_sorted) // 15)
    sample = clean_rows_sorted[::step][:15]

    print()
    print("15-row sample from clean therapeutic set (every ~N-th alphabetically):")
    hdr = ["drug_name", "therapeutic_min_mg_L", "therapeutic_max_mg_L",
           "toxic_min_mg_L", "half_life_h", "data_quality_flag"]
    col_w = [38, 22, 22, 15, 10, 30]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
    print(fmt.format(*hdr))
    print("-" * sum(col_w + [2 * len(col_w)]))
    for r in sample:
        print(fmt.format(
            r["drug_name"][:col_w[0]],
            r["therapeutic_min_mg_L"][:col_w[1]],
            r["therapeutic_max_mg_L"][:col_w[2]],
            (r.get("toxic_min_mg_L") or "")[:col_w[3]],
            (r.get("half_life_h") or "")[:col_w[4]],
            (r.get("data_quality_flag") or "")[:col_w[5]],
        ))

    print()
    print(f"Output -> {_OUT_CSV}")


if __name__ == "__main__":
    run()
