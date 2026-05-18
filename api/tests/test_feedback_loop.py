"""Feedback CSV logging and fine-tune CLI guard."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]
PROJ_ROOT = API_ROOT.parent


def test_record_feedback_writes_rows(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DLPBPK_FEEDBACK_LOG", str(tmp_path / "fb.csv"))
    sys.path.insert(0, str(PROJ_ROOT))
    sys.path.insert(0, str(API_ROOT))
    from app.services.feedback_service import record_feedback, summarize_feedback

    record_feedback(
        {"drug": "theophylline", "patient_id": 999, "weight_kg": 70, "dose_mg": 200, "age_years": 40, "sex": 0.0},
        predicted_concentrations=[1.0, 1.2],
        observed_concentrations=[1.05, 1.15],
        t_hours=[0.0, 1.0],
        source="pytest",
    )
    s = summarize_feedback("theophylline")
    assert s["n_rows"] == 2
    with open(tmp_path / "fb.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2


def test_finetune_insufficient_feedback_exits_cleanly(tmp_path, monkeypatch) -> None:
    """``finetune_from_feedback`` prints FAILED but does not raise when data are sparse."""
    fb = tmp_path / "tiny.csv"
    fb.write_text("timestamp,drug,patient_id,weight_kg,dose_mg,age_years,sex,t_hours,predicted_mg_per_L,observed_mg_per_L,source\n", encoding="utf-8")
    cmd = [
        sys.executable,
        str(PROJ_ROOT / "experiments" / "training" / "finetune_from_feedback.py"),
        "--drug",
        "theophylline",
        "--min-feedback-points",
        "9999",
        "--feedback-csv",
        str(fb),
    ]
    proc = subprocess.run(cmd, cwd=str(PROJ_ROOT), capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0
    assert "FAILED finetune_from_feedback" in proc.stdout or "FAILED finetune_from_feedback" in proc.stderr
