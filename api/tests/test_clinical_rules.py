"""Clinical interpretation layer for /recommend."""

from __future__ import annotations

import sys
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]
PROJ_ROOT = API_ROOT.parent
sys.path.insert(0, str(PROJ_ROOT))
sys.path.insert(0, str(API_ROOT))

from app.services.clinical_rules_service import dose_adjustment_rule, explain_for_recommendation


def test_high_cmax_triggers_dose_reduction_heuristic() -> None:
    r = dose_adjustment_rule("theophylline", 18.5, 90.0)
    assert r.evidence_tier == "extrapolation"
    assert r.adjustment_pct is not None and r.adjustment_pct > 0
    assert "mg/L" in r.reasoning_text


def test_in_band_literature_tier() -> None:
    r = dose_adjustment_rule("theophylline", 10.0, 80.0)
    assert r.evidence_tier == "literature_band"
    assert "within" in r.reasoning_text.lower() or "band" in r.reasoning_text.lower()


def test_context_label_preserved() -> None:
    r = explain_for_recommendation("warfarin", 1.0, 50.0, "baseline")
    assert r.context == "baseline"
