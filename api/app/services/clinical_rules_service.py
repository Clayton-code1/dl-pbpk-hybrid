"""Literature-band–anchored clinical interpretation for dose recommendations."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.reference_pk import REFERENCE_PK_DATA  # noqa: E402


@dataclass(frozen=True)
class ClinicalReasoning:
    """Structured clinical narrative tier for examiner-facing API payloads."""

    context: str
    adjustment_pct: float | None
    reasoning_text: str
    evidence_tier: str


def reference_table(drug: str) -> dict:
    key = drug.strip().lower()
    if key not in REFERENCE_PK_DATA:
        raise KeyError(f"No reference PK row for drug={drug!r}")
    return REFERENCE_PK_DATA[key]


def _safe_pct_change(value: float, bound: float) -> float:
    if bound <= 0:
        return 0.0
    return abs(value - bound) / bound * 100.0


def dose_adjustment_rule(
    drug: str,
    predicted_cmax_mg_L: float,
    predicted_auc_mg_h_L: float,
) -> ClinicalReasoning:
    """Map predicted exposure to a tier + approximate dose adjustment heuristic."""
    ref = reference_table(drug)
    t_lo = float(ref["therapeutic_min_mg_L"])
    t_hi = float(ref["therapeutic_max_mg_L"])
    ref_auc = float(ref["AUC_mg_h_L"])

    evidence_tier = "literature_band"
    if predicted_cmax_mg_L < t_lo:
        margin = t_lo * 0.5
        if predicted_cmax_mg_L < margin:
            evidence_tier = "out_of_scope"
        else:
            evidence_tier = "extrapolation"
    elif predicted_cmax_mg_L > t_hi:
        if predicted_cmax_mg_L > t_hi * 1.5:
            evidence_tier = "out_of_scope"
        else:
            evidence_tier = "extrapolation"

    auc_high = ref_auc * 2.0
    auc_low = ref_auc * 0.25
    if predicted_auc_mg_h_L > auc_high or predicted_auc_mg_h_L < auc_low:
        if evidence_tier == "literature_band":
            evidence_tier = "extrapolation"

    adjustment_pct: float | None = None
    if predicted_cmax_mg_L > t_hi > 0:
        excess_pct = _safe_pct_change(predicted_cmax_mg_L, t_hi)
        adjustment_pct = round(min(90.0, max(5.0, excess_pct * 0.9)), 1)
        reasoning = (
            f"Predicted Cmax {predicted_cmax_mg_L:.2f} mg/L exceeds upper therapeutic bound "
            f"{t_hi:.2f} mg/L by {excess_pct:.1f}%; suggest approximate dose reduction of "
            f"{adjustment_pct:.1f}% relative to the simulated regimen. "
            f"Predicted AUC {predicted_auc_mg_h_L:.2f} mg·h/L (literature order-of-magnitude ~{ref_auc:.1f}). "
            f"Reference: {ref['reference']}."
        )
    elif predicted_cmax_mg_L < t_lo > 0:
        deficit_pct = _safe_pct_change(predicted_cmax_mg_L, t_lo)
        adjustment_pct = round(-min(90.0, max(5.0, deficit_pct * 0.9)), 1)
        reasoning = (
            f"Predicted Cmax {predicted_cmax_mg_L:.2f} mg/L is below lower therapeutic bound "
            f"{t_lo:.2f} mg/L (~{deficit_pct:.1f}% low); upward titration on the order of "
            f"{abs(adjustment_pct):.1f}% may be considered under monitoring — algorithmic suggestion only. "
            f"Reference: {ref['reference']}."
        )
    else:
        reasoning = (
            f"Predicted Cmax {predicted_cmax_mg_L:.2f} mg/L lies within the literature therapeutic band "
            f"[{t_lo:.2f}, {t_hi:.2f}] mg/L (predicted AUC {predicted_auc_mg_h_L:.2f} mg·h/L). "
            f"Reference: {ref['reference']}."
        )

    if evidence_tier == "out_of_scope":
        reasoning += " Evidence tier: exposure far from published therapeutic window — recommend specialist review."

    return ClinicalReasoning(
        context="exposure_check",
        adjustment_pct=adjustment_pct,
        reasoning_text=reasoning,
        evidence_tier=evidence_tier,
    )


def explain_for_recommendation(
    drug: str,
    predicted_cmax_mg_L: float,
    predicted_auc_mg_h_L: float,
    context_label: str,
) -> ClinicalReasoning:
    """Wrap :func:`dose_adjustment_rule` with API context (baseline vs recommended)."""
    base = dose_adjustment_rule(drug, predicted_cmax_mg_L, predicted_auc_mg_h_L)
    return ClinicalReasoning(
        context=context_label,
        adjustment_pct=base.adjustment_pct,
        reasoning_text=base.reasoning_text,
        evidence_tier=base.evidence_tier,
    )
