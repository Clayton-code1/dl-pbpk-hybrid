# Phase 3.1 — Literature-backed therapeutic concentration bands

Peak plasma concentration **C** (mg/L) windows from `REFERENCE_PK_DATA` (`therapeutic_min_mg_L`, `therapeutic_max_mg_L`). These replace heuristic API references for drug-specific inference.

| Drug | min (mg/L) | max (mg/L) | Clinical reference |
|---|---:|---:|---|
| theophylline | 5.0 | 15.0 | Hendeles & Weinberger, 1982 |
| warfarin | 0.5 | 3.0 | Holford, 1986 |
| midazolam | 0.04 | 0.3 | Smith et al., 1981 |
| caffeine | 5.0 | 20.0 | Arnaud, 1993 |
| acetaminophen | 5.0 | 20.0 | Prescott, 1980 |
| digoxin | 0.0005 | 0.002 | Reuning et al., 1973 |

## API integration

`api/app/services/risk_service.py` exposes `literature_therapeutic_window()` and enriches `assess_risk(..., drug=...)` with `literature_status` when a canonical drug name is provided.
