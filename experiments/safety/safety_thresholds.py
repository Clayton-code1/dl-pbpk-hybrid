"""Phase 3.1 — document literature-backed peak concentration windows.

Updates API risk scoring to expose literature windows when a drug key is known.
Generates ``experiments/results/phase3_safety_thresholds.md``.

    python -m experiments.safety.safety_thresholds
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.config import (  # noqa: E402
    DRUGS,
    RESULTS_DIR,
    ensure_dirs,
    get_logger,
)
from experiments.reference_pk import REFERENCE_PK_DATA  # noqa: E402

LOGGER = get_logger("phase3.safety", "phase3_safety_thresholds.log")


def main() -> int:
    ensure_dirs()
    lines = [
        "# Phase 3.1 — Literature-backed therapeutic concentration bands",
        "",
        "Peak plasma concentration **C** (mg/L) windows from `REFERENCE_PK_DATA` "
        "(`therapeutic_min_mg_L`, `therapeutic_max_mg_L`). "
        "These replace heuristic API references for drug-specific inference.",
        "",
        "| Drug | min (mg/L) | max (mg/L) | Clinical reference |",
        "|---|---:|---:|---|",
    ]
    for d in DRUGS:
        r = REFERENCE_PK_DATA[d]
        lines.append(
            f"| {d} | {r['therapeutic_min_mg_L']} | {r['therapeutic_max_mg_L']} | "
            f"{r['reference']} |"
        )

    lines.extend(
        [
            "",
            "## API integration",
            "",
            "`api/app/services/risk_service.py` exposes `literature_therapeutic_window()` "
            "and enriches `assess_risk(..., drug=...)` with `literature_status` when a "
            "canonical drug name is provided.",
        ]
    )

    out = RESULTS_DIR / "phase3_safety_thresholds.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
