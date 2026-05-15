"""
Standalone evaluation script for the DL-PBPK prototype.

Calls the live /predict/v2 API with 5 fixed Theophylline dose scenarios,
collects PK metrics and safety, and saves results to CSV, Markdown, and JSON
in artifacts/evaluation/theophylline_progress_review/.

Usage (from repo root = dl-pbpk-hybrid):
    python scripts/evaluation/run_theophylline_eval.py

Requires: requests, pandas. Install with: pip install requests pandas
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: 'requests' is required. Install with: pip install requests", file=sys.stderr)
    sys.exit(1)
try:
    import pandas as pd
except ImportError:
    print("Error: 'pandas' is required. Install with: pip install pandas", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE = "http://127.0.0.1:8000"
PREDICT_V2_URL = f"{API_BASE}/predict/v2"

# Fixed test cases: (compound, weight_kg, dose_mg, route)
TEST_CASES = [
    ("Theophylline", 70, 50, "oral"),
    ("Theophylline", 70, 100, "oral"),
    ("Theophylline", 70, 150, "oral"),
    ("Theophylline", 70, 200, "oral"),
    ("Theophylline", 70, 300, "oral"),
]

# Output column order (presentation-ready names)
OUTPUT_COLUMNS = [
    "timestamp",
    "dose_mg",
    "weight_kg",
    "route",
    "safety_status",
    "risk_score",
    "Cmax_ng_per_mL",
    "Tmax_h",
    "AUC_ng_h_per_mL",
    "half_life_h",
    "clearance_L_per_h",
    "volume_of_distribution_L",
    "model_used",
    "model_version",
    "error",
]


def get_repo_root() -> Path:
    """Repo root = parent of parent of script dir (script in scripts/evaluation/)."""
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent.parent


def get_output_dir() -> Path:
    return get_repo_root() / "artifacts" / "evaluation" / "theophylline_progress_review"


def build_request(compound: str, weight_kg: float, dose_mg: float, route: str) -> dict:
    return {
        "patient": {"weight_kg": weight_kg, "compound_name": compound},
        "regimen": [{"time_hr": 0.0, "dose_mg": dose_mg, "route": route}],
        "horizon_hr": 48.0,
        "dt_min": 5.0,
    }


def run_one(
    compound: str, weight_kg: float, dose_mg: float, route: str, timestamp: str
) -> dict:
    """Call /predict/v2 for one test case. Return a flat row dict or row with error."""
    row = {
        "timestamp": timestamp,
        "dose_mg": dose_mg,
        "weight_kg": weight_kg,
        "route": route,
        "safety_status": "",
        "risk_score": None,
        "Cmax_ng_per_mL": None,
        "Tmax_h": None,
        "AUC_ng_h_per_mL": None,
        "half_life_h": None,
        "clearance_L_per_h": None,
        "volume_of_distribution_L": None,
        "model_used": "",
        "model_version": "",
        "error": "",
    }
    try:
        resp = requests.post(
            PREDICT_V2_URL,
            json=build_request(compound, weight_kg, dose_mg, route),
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError as e:
        row["error"] = f"Connection error: {e}"
        return row
    except requests.exceptions.Timeout as e:
        row["error"] = f"Timeout: {e}"
        return row
    except requests.exceptions.HTTPError as e:
        row["error"] = f"HTTP {e.response.status_code}: {e.response.text[:200] if e.response.text else str(e)}"
        return row
    except Exception as e:
        row["error"] = str(e)
        return row

    # Extract fields
    try:
        safety = data.get("safety") or {}
        row["safety_status"] = "safe" if safety.get("is_safe") else "unsafe"
        row["risk_score"] = safety.get("risk_score")
    except Exception:
        row["safety_status"] = "unknown"
        row["error"] = (row["error"] or "") + "; extract safety failed"

    try:
        pk = data.get("pk_metrics") or {}
        row["Cmax_ng_per_mL"] = pk.get("cmax_ng_ml")
        row["Tmax_h"] = pk.get("tmax_h")
        row["AUC_ng_h_per_mL"] = pk.get("auc_0_inf")
        row["half_life_h"] = pk.get("half_life_h")
        row["clearance_L_per_h"] = pk.get("clearance_l_h")
        row["volume_of_distribution_L"] = pk.get("vd_l")
    except Exception:
        if not row["error"]:
            row["error"] = "extract pk_metrics failed"

    try:
        model = data.get("model") or {}
        row["model_used"] = model.get("model_used") or ""
        row["model_version"] = model.get("version") or ""
    except Exception:
        pass

    return row


def main() -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    output_dir = get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for compound, weight_kg, dose_mg, route in TEST_CASES:
        print(f"  Running: {compound} {dose_mg} mg, {weight_kg} kg, {route}...", flush=True)
        row = run_one(compound, weight_kg, dose_mg, route, ts)
        rows.append(row)

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    # Paths
    csv_path = output_dir / "theophylline_results.csv"
    md_path = output_dir / "theophylline_results.md"
    json_path = output_dir / "theophylline_results.json"

    df.to_csv(csv_path, index=False, encoding="utf-8")
    _write_md_manually(df, md_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    # Console summary
    print()
    print("Results summary:")
    print(df.to_string(index=False))
    print()
    print("Output file locations:")
    print(f"  CSV:  {csv_path}")
    print(f"  MD:   {md_path}")
    print(f"  JSON: {json_path}")
    print()
    n_ok = sum(1 for r in rows if not r.get("error"))
    print(f"Completed: {n_ok}/{len(rows)} test cases without error.")


def _write_md_manually(df: pd.DataFrame, path: Path) -> None:
    """Write a simple Markdown table if pandas has no to_markdown (older pandas)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(df.columns) + " |\n")
        f.write("| " + " | ".join("---" for _ in df.columns) + " |\n")
        for _, r in df.iterrows():
            f.write("| " + " | ".join(_md_cell(r[c]) for c in df.columns) + " |\n")


def _md_cell(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x)


if __name__ == "__main__":
    main()
