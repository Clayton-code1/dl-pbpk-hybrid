"""Therapeutic concentration bands from ``REFERENCE_PK_DATA`` (mg/L)."""

from __future__ import annotations

from experiments.reference_pk import REFERENCE_PK_DATA


def therapeutic_window_mg_L(drug: str) -> tuple[float, float]:
    r = REFERENCE_PK_DATA[drug]
    return float(r["therapeutic_min_mg_L"]), float(r["therapeutic_max_mg_L"])


def classify_peak_mg_L(drug: str, cmax_mg_L: float) -> str:
    lo, hi = therapeutic_window_mg_L(drug)
    if cmax_mg_L < lo:
        return "below_therapeutic"
    if cmax_mg_L > hi:
        return "above_therapeutic"
    return "therapeutic"
