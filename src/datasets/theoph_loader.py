"""Load and normalise the Theophylline PK dataset from CSV.

Expected CSV columns (case-insensitive):
    rownames, Subject, Wt, Dose, Time, conc

Returns a list of per-subject dicts ready for downstream modelling or
serialisation to JSON.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CSV = _PROJECT_ROOT / "data" / "raw" / "theoph" / "theoph.csv"

_COLUMN_MAP = {
    "subject": "subject_id",
    "wt": "weight_kg",
    "dose": "dose_mgkg",
    "time": "time_hr",
    "conc": "conc_mgL",
}


def load_theoph(csv_path: Path | str | None = None) -> list[dict[str, Any]]:
    """Read theoph.csv and return a list of normalised subject records.

    Parameters
    ----------
    csv_path : path-like, optional
        Override the default CSV location.

    Returns
    -------
    list[dict]
        Each dict contains subject_id, weight_kg, dose_mgkg, dose_mg,
        route, times_hr, concentration, and units.
    """
    path = Path(csv_path) if csv_path is not None else _DEFAULT_CSV
    df = pd.read_csv(path)

    # Normalise column names to lowercase for robust matching
    df.columns = df.columns.str.strip().str.lower()

    # Drop the rownames index column if present
    if "rownames" in df.columns:
        df = df.drop(columns=["rownames"])

    # Rename to canonical names
    df = df.rename(columns=_COLUMN_MAP)

    # Derive total dose in mg
    df["dose_mg"] = df["dose_mgkg"] * df["weight_kg"]

    # Sort within each subject
    df = df.sort_values(["subject_id", "time_hr"]).reset_index(drop=True)

    subjects: list[dict[str, Any]] = []
    for sid, grp in df.groupby("subject_id", sort=True):
        row = grp.iloc[0]
        subjects.append(
            {
                "subject_id": str(int(sid)),
                "weight_kg": float(row["weight_kg"]),
                "dose_mgkg": float(row["dose_mgkg"]),
                "dose_mg": round(float(row["dose_mg"]), 2),
                "route": "oral",
                "times_hr": grp["time_hr"].tolist(),
                "concentration": grp["conc_mgL"].tolist(),
                "units": "mg/L",
            }
        )

    return subjects
