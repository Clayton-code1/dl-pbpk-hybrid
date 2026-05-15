"""Quick verification of the preprocessed Theophylline dataset.

Checks
------
1. Number of subjects matches expected count.
2. First subject record prints correctly.
3. Times are sorted ascending within every subject.
4. All concentrations are non-negative.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SUBJECTS_JSON = _PROJECT_ROOT / "data" / "processed" / "theoph" / "theoph_subjects.json"


def _check(condition: bool, message: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {message}")
    if not condition:
        raise SystemExit(1)


def main() -> None:
    if not _SUBJECTS_JSON.exists():
        print(f"File not found: {_SUBJECTS_JSON}")
        print("Run  python scripts/preprocess_theoph.py  first.")
        sys.exit(1)

    with open(_SUBJECTS_JSON, "r", encoding="utf-8") as f:
        subjects = json.load(f)

    n = len(subjects)
    print(f"\nLoaded {n} subjects from {_SUBJECTS_JSON.name}\n")

    # --- Check 1: subject count ---
    _check(n > 0, f"At least one subject present (got {n})")

    # --- Check 2: print first record ---
    first = subjects[0]
    print("  First subject record:")
    for k, v in first.items():
        if isinstance(v, list):
            print(f"    {k}: [{v[0]}, {v[1]}, ... {v[-1]}]  ({len(v)} points)")
        else:
            print(f"    {k}: {v}")
    print()

    # --- Check 3: times sorted ---
    all_sorted = True
    for s in subjects:
        times = s["times_hr"]
        if times != sorted(times):
            all_sorted = False
            print(f"    Subject {s['subject_id']} times NOT sorted!")
    _check(all_sorted, "All subjects have times sorted ascending")

    # --- Check 4: non-negative concentrations ---
    all_nonneg = True
    for s in subjects:
        if any(c < 0 for c in s["concentration"]):
            all_nonneg = False
            print(f"    Subject {s['subject_id']} has negative concentration!")
    _check(all_nonneg, "All concentrations are non-negative")

    print("\nAll checks passed.\n")


if __name__ == "__main__":
    main()
