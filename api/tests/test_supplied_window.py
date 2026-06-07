"""Tests for the caller-supplied therapeutic-window path in assess_risk().

All tests call assess_risk() directly so they run without model artifacts.

Window used throughout: 10.0–20.0 mg/L.
Helper _ng(mg_L) converts to the ng/mL unit that assess_risk() expects.
"""

from __future__ import annotations

import pytest

from app.services.risk_service import assess_risk

_WINDOW = {"therapeutic_min_mg_L": 10.0, "therapeutic_max_mg_L": 20.0}


def _ng(mg_L: float) -> float:
    """Convert mg/L to ng/mL (×1000)."""
    return mg_L * 1000.0


# ---------------------------------------------------------------------------
# 1. In-window → safe
# ---------------------------------------------------------------------------

def test_supplied_window_in_range_is_safe():
    """Cmax inside window → is_safe=True, zone='therapeutic'."""
    result = assess_risk(_ng(15.0), 0.0, **_WINDOW)
    assert result["is_safe"] is True
    assert result["supplied_window"]["zone"] == "therapeutic"
    assert result["supplied_window"]["therapeutic_min_mg_L"] == 10.0
    assert result["supplied_window"]["therapeutic_max_mg_L"] == 20.0


# ---------------------------------------------------------------------------
# 2. Above window → unsafe
# ---------------------------------------------------------------------------

def test_supplied_window_above_max_is_unsafe():
    """Cmax above window → is_safe=False, zone='above_therapeutic'."""
    result = assess_risk(_ng(25.0), 0.0, **_WINDOW)
    assert result["is_safe"] is False
    assert result["supplied_window"]["zone"] == "above_therapeutic"


# ---------------------------------------------------------------------------
# 3. Below window → unsafe
# ---------------------------------------------------------------------------

def test_supplied_window_below_min_is_unsafe():
    """Cmax below window → is_safe=False, zone='below_therapeutic'."""
    result = assess_risk(_ng(4.8), 0.0, **_WINDOW)
    assert result["is_safe"] is False
    assert result["supplied_window"]["zone"] == "below_therapeutic"


# ---------------------------------------------------------------------------
# 4. No window supplied → generic sigmoid fallback, no supplied_window key
# ---------------------------------------------------------------------------

def test_no_window_uses_generic_fallback():
    """Unknown drug with no window → generic path, supplied_window absent."""
    result = assess_risk(_ng(15.0), 200_000.0)
    # Generic path never populates supplied_window
    assert "supplied_window" not in result
    # is_safe is determined by the sigmoid (should be False at these inflated values)
    assert isinstance(result["is_safe"], bool)
    assert "risk_score" in result


# ---------------------------------------------------------------------------
# 5. Panel drug via drug= kwarg → original path, supplied_window absent
# ---------------------------------------------------------------------------

def test_panel_drug_kwarg_uses_original_path():
    """assess_risk called with drug='theophylline' (no window) → original path."""
    result = assess_risk(_ng(15.0), 200_000.0, drug="theophylline")
    assert "supplied_window" not in result
    assert isinstance(result["is_safe"], bool)


# ---------------------------------------------------------------------------
# 6. Boundary: Cmax exactly at therapeutic_min → is_safe=True
# ---------------------------------------------------------------------------

def test_boundary_cmax_equals_min_is_safe():
    """Cmax exactly equal to therapeutic_min (10.0 mg/L) → is_safe=True."""
    result = assess_risk(_ng(10.0), 0.0, **_WINDOW)
    assert result["is_safe"] is True
    assert result["supplied_window"]["zone"] == "therapeutic"


# ---------------------------------------------------------------------------
# 7. Boundary: Cmax exactly at therapeutic_max → is_safe=True
# ---------------------------------------------------------------------------

def test_boundary_cmax_equals_max_is_safe():
    """Cmax exactly equal to therapeutic_max (20.0 mg/L) → is_safe=True."""
    result = assess_risk(_ng(20.0), 0.0, **_WINDOW)
    assert result["is_safe"] is True
    assert result["supplied_window"]["zone"] == "therapeutic"
