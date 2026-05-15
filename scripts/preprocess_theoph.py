"""Preprocess the Theophylline dataset and write JSON outputs.

Outputs
-------
data/processed/theoph/theoph_subjects.json
    List of per-subject PK records.

data/processed/theoph/theoph_summary.json
    Aggregate summary statistics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running from the repository root (python scripts/preprocess_theoph.py)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.datasets.theoph_loader import load_theoph  # noqa: E402

_OUTPUT_DIR = _PROJECT_ROOT / "data" / "processed" / "theoph"


def main() -> None:
    print("Loading Theophylline CSV ...")
    subjects = load_theoph()
    print(f"  -> {len(subjects)} subjects loaded")

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- subjects JSON ---
    subjects_path = _OUTPUT_DIR / "theoph_subjects.json"
    with open(subjects_path, "w", encoding="utf-8") as f:
        json.dump(subjects, f, indent=2)
    print(f"  -> Wrote {subjects_path}")

    # --- summary JSON ---
    all_times: list[float] = []
    all_concs: list[float] = []
    weights: list[float] = []
    doses: list[float] = []

    for s in subjects:
        all_times.extend(s["times_hr"])
        all_concs.extend(s["concentration"])
        weights.append(s["weight_kg"])
        doses.append(s["dose_mgkg"])

    summary = {
        "n_subjects": len(subjects),
        "n_points_total": len(all_times),
        "time_min": min(all_times),
        "time_max": max(all_times),
        "conc_min": min(all_concs),
        "conc_max": max(all_concs),
        "mean_weight_kg": round(sum(weights) / len(weights), 2),
        "mean_dose_mgkg": round(sum(doses) / len(doses), 3),
    }

    summary_path = _OUTPUT_DIR / "theoph_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  -> Wrote {summary_path}")

    print("\nSummary")
    print("-------")
    for k, v in summary.items():
        print(f"  {k:20s}: {v}")

    print("\nDone.")


if __name__ == "__main__":
    main()
