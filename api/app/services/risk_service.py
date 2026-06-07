"""Calibrated risk scoring service for dose safety assessment.

risk_score = sigmoid(a * log_cmax_ratio + b * log_auc_ratio + c)

where:
    log_cmax_ratio = log(cmax + eps) - log(cmax_ref)
    log_auc_ratio  = log(auc  + eps) - log(auc_ref)

Calibration parameters are loaded from and persisted to
api/app/state/model_state.json.
"""

from __future__ import annotations

import json
import math
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("uvicorn.error")

_STATE_PATH = Path(__file__).resolve().parent.parent / "state" / "model_state.json"
_EPS = 1e-3

_DEFAULTS = {
    "a": 2.0,
    "b": 2.0,
    "c": 0.0,
    "cmax_ref": 2000.0,    # ng/mL  — fallback when drug is unknown
    "auc_ref": 30000.0,    # ng*h/mL  — fallback when drug is unknown
    "threshold": 0.5,
}


def _load_state() -> dict[str, Any]:
    if _STATE_PATH.exists():
        with open(_STATE_PATH, "r") as f:
            return json.load(f)
    return {
        "calibration": dict(_DEFAULTS),
        "update_flag": False,
        "updated_at": None,
    }


def _save_state(state: dict[str, Any]) -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _project_root() -> Path:
    """``dl-pbpk-hybrid`` (parent of ``api``)."""
    return Path(__file__).resolve().parents[3]


def literature_therapeutic_window(drug: str) -> tuple[float, float, str] | None:
    """Return ``(Cmin_mg_L, Cmax_mg_L, reference)`` from Phase 3 literature table."""
    try:
        import sys

        root = _project_root()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from experiments.reference_pk import REFERENCE_PK_DATA  # type: ignore

        if drug not in REFERENCE_PK_DATA:
            return None
        r = REFERENCE_PK_DATA[drug]
        return (
            float(r["therapeutic_min_mg_L"]),
            float(r["therapeutic_max_mg_L"]),
            str(r["reference"]),
        )
    except Exception:  # pragma: no cover
        return None


def _literature_peak_status(drug: str, cmax_mg_L: float) -> dict[str, Any] | None:
    band = literature_therapeutic_window(drug)
    if band is None:
        return None
    lo, hi, ref = band
    if cmax_mg_L < lo:
        zone = "below_therapeutic"
    elif cmax_mg_L > hi:
        zone = "above_therapeutic"
    else:
        zone = "therapeutic"
    return {
        "therapeutic_min_mg_L": lo,
        "therapeutic_max_mg_L": hi,
        "reference": ref,
        "literature_zone": zone,
    }


def assess_risk(
    cmax_ng_ml: float,
    auc_ng_h_ml: float,
    *,
    drug: str | None = None,
    therapeutic_min_mg_L: float | None = None,
    therapeutic_max_mg_L: float | None = None,
) -> dict[str, Any]:
    """Return risk assessment dict with is_safe, risk_score, reason.

    When *both* therapeutic_min_mg_L and therapeutic_max_mg_L are supplied the
    safety decision is made by comparing predicted Cmax against that window.
    The sigmoid risk_score is still computed and returned for reference.
    This path is only reached for non-panel drugs (panel drugs never supply a
    window — main.py guards this with an elif so panel behaviour is unchanged).
    """
    state = _load_state()
    cal = state.get("calibration", _DEFAULTS)

    a = cal.get("a", _DEFAULTS["a"])
    b = cal.get("b", _DEFAULTS["b"])
    c = cal.get("c", _DEFAULTS["c"])
    cmax_ref = cal.get("cmax_ref", _DEFAULTS["cmax_ref"])
    auc_ref = cal.get("auc_ref", _DEFAULTS["auc_ref"])
    threshold = cal.get("threshold", _DEFAULTS["threshold"])

    log_cmax_r = math.log(cmax_ng_ml + _EPS) - math.log(cmax_ref)
    log_auc_r  = math.log(auc_ng_h_ml + _EPS) - math.log(auc_ref)
    score = _sigmoid(a * log_cmax_r + b * log_auc_r + c)

    use_supplied_window = (
        therapeutic_min_mg_L is not None and therapeutic_max_mg_L is not None
    )

    if use_supplied_window:
        # Safety gate: is predicted Cmax within the caller-supplied window?
        cmax_mg_L = cmax_ng_ml / 1000.0
        lo = float(therapeutic_min_mg_L)  # type: ignore[arg-type]
        hi = float(therapeutic_max_mg_L)  # type: ignore[arg-type]

        if cmax_mg_L < lo:
            zone = "below_therapeutic"
            is_safe = False
            reason = (
                f"Predicted Cmax ({cmax_mg_L:.3g} mg/L) is below the supplied "
                f"therapeutic minimum ({lo:.3g} mg/L)."
            )
        elif cmax_mg_L > hi:
            zone = "above_therapeutic"
            is_safe = False
            reason = (
                f"Predicted Cmax ({cmax_mg_L:.3g} mg/L) exceeds the supplied "
                f"therapeutic maximum ({hi:.3g} mg/L)."
            )
        else:
            zone = "therapeutic"
            is_safe = True
            reason = (
                f"Predicted Cmax ({cmax_mg_L:.3g} mg/L) is within the supplied "
                f"therapeutic window ({lo:.3g}–{hi:.3g} mg/L)."
            )

        out: dict[str, Any] = {
            "is_safe": is_safe,
            "risk_score": round(score, 4),
            "reason": reason,
            "supplied_window": {
                "therapeutic_min_mg_L": lo,
                "therapeutic_max_mg_L": hi,
                "zone": zone,
            },
        }
        return out

    # --- Original path: generic calibration fallback (panel drugs and unknown
    #     drugs without a supplied window reach here unchanged) ---
    is_safe = score < threshold

    cmax_high = log_cmax_r > 0
    auc_high  = log_auc_r > 0
    if not cmax_high and not auc_high:
        reason = "Both Cmax and AUC are within acceptable ranges."
    elif cmax_high and auc_high:
        reason = "Both Cmax and AUC exceed reference thresholds."
    elif cmax_high:
        reason = f"Cmax ({cmax_ng_ml:.0f} ng/mL) exceeds reference ({cmax_ref:.0f} ng/mL)."
    else:
        reason = f"AUC ({auc_ng_h_ml:.0f} ng*h/mL) exceeds reference ({auc_ref:.0f} ng*h/mL)."

    out = {
        "is_safe": is_safe,
        "risk_score": round(score, 4),
        "reason": reason,
    }
    if drug is not None:
        cmax_mg_L = cmax_ng_ml / 1000.0
        lit = _literature_peak_status(drug, cmax_mg_L)
        if lit is not None:
            out["literature"] = lit
    return out


def get_model_state() -> dict[str, Any]:
    return _load_state()


def update_calibration(feedback_label: str) -> dict[str, Any]:
    """Nudge calibration based on clinician feedback.

    'unsafe' -> tighten (increase a/b, lower threshold slightly)
    'safe'   -> relax  (decrease a/b, raise threshold slightly)
    """
    state = _load_state()
    cal = state.get("calibration", dict(_DEFAULTS))
    step = 0.05

    if feedback_label.lower() == "unsafe":
        cal["a"] = round(cal.get("a", 2.0) + step, 4)
        cal["b"] = round(cal.get("b", 2.0) + step, 4)
        cal["threshold"] = round(max(0.1, cal.get("threshold", 0.5) - step / 2), 4)
    elif feedback_label.lower() == "safe":
        cal["a"] = round(max(0.1, cal.get("a", 2.0) - step), 4)
        cal["b"] = round(max(0.1, cal.get("b", 2.0) - step), 4)
        cal["threshold"] = round(min(0.9, cal.get("threshold", 0.5) + step / 2), 4)

    state["calibration"] = cal
    state["update_flag"] = True
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    _save_state(state)

    return state
