"""Feedback log for continual-learning demos (PK observations vs model predictions)."""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_FEEDBACK_DIR = _PROJECT_ROOT / "experiments" / "data" / "feedback"

_FEEDBACK_COLUMNS = [
    "timestamp",
    "drug",
    "patient_id",
    "weight_kg",
    "dose_mg",
    "age_years",
    "sex",
    "t_hours",
    "predicted_mg_per_L",
    "observed_mg_per_L",
    "source",
]


def feedback_log_path() -> Path:
    """CSV path; override with env ``DLPBPK_FEEDBACK_LOG`` for tests."""
    env = os.environ.get("DLPBPK_FEEDBACK_LOG")
    if env:
        return Path(env)
    return _FEEDBACK_DIR / "feedback_log.csv"


def record_feedback(
    patient_record: dict[str, Any],
    predicted_concentrations: list[float],
    observed_concentrations: list[float],
    t_hours: list[float],
    source: str,
) -> None:
    """Append one CSV row per time point (aligned lists)."""
    drug = str(patient_record["drug"])
    pid = int(patient_record["patient_id"])
    w = float(patient_record["weight_kg"])
    dose = float(patient_record["dose_mg"])
    age = float(patient_record["age_years"])
    sex = float(patient_record["sex"])

    path = feedback_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    ts = datetime.now(timezone.utc).isoformat()

    with open(path, "a", newline="", encoding="utf-8") as f:
        wtr = csv.writer(f)
        if write_header:
            wtr.writerow(_FEEDBACK_COLUMNS)
        for t_h, pred, obs in zip(t_hours, predicted_concentrations, observed_concentrations, strict=True):
            wtr.writerow([ts, drug, pid, w, dose, age, sex, t_h, pred, obs, source])


def summarize_feedback(drug: str | None = None) -> dict[str, Any]:
    """Return counts and mean absolute error (pred vs obs) from the log."""
    path = feedback_log_path()
    if not path.exists():
        return {"n_rows": 0, "n_patients": 0, "mae_mg_L": None, "by_drug": {}}

    import csv as _csv

    rows: list[dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        rdr = _csv.DictReader(f)
        for row in rdr:
            if drug and row.get("drug", "").lower() != drug.lower():
                continue
            rows.append(row)

    if not rows:
        return {"n_rows": 0, "n_patients": 0, "mae_mg_L": None, "by_drug": {}}

    abs_err: list[float] = []
    patients: set[str] = set()
    by_drug: dict[str, int] = {}
    for row in rows:
        by_drug[row["drug"]] = by_drug.get(row["drug"], 0) + 1
        patients.add(f'{row["drug"]}:{row["patient_id"]}')
        try:
            p = float(row["predicted_mg_per_L"])
            o = float(row["observed_mg_per_L"])
            abs_err.append(abs(p - o))
        except (KeyError, ValueError):
            continue

    mae = float(sum(abs_err) / len(abs_err)) if abs_err else None
    return {
        "n_rows": len(rows),
        "n_patients": len(patients),
        "mae_mg_L": mae,
        "by_drug": by_drug,
    }
