# Objectives Elevation — Verification Log

Generated: 2026-05-18. Project root: `dl-pbpk-hybrid/`.

| Step | Command / check | Result |
|------|-----------------|--------|
| 1 | `python experiments/explainability/graph_shap_lite.py --dry-run` | **PASS** — prints planned drug list and exits 0 |
| 2 | `python experiments/explainability/graph_shap_lite.py --drugs theophylline` | **PASS** — `experiments/results/phase3b_graph_explainability.csv` updated; `experiments/plots/graph_explainability_theophylline.png` created |
| 3 | `python experiments/explainability/error_attribution.py --drugs theophylline` | **PASS** (after CSV column fix; superseded by full-panel run below which refreshed all six drugs) |
| 4 | `python experiments/training/finetune_from_feedback.py --drug theophylline --dry-run` | **PASS** |
| 5 | `python experiments/training/finetune_from_feedback.py --drug theophylline` | **PASS** — `artifacts/models/hybrid_gnn_pbpk_theophylline_v_finetuned/` with `finetune_report.json` |
| 6 | `pytest api/tests/test_xai_real_backgrounds.py -v` | **PASS** (7 tests total with other new modules) |
| 7 | Paths in `OBJECTIVES_COMPLETION_REPORT.md` | **PASS** — spot-checked file paths exist on disk after this pass |

## Supplemental runs (not all in original checklist)

- **PASS** — `python experiments/explainability/graph_shap_lite.py` (all six training drugs) — full CSV + `graph_explainability_grid.png`
- **PASS** — `python experiments/explainability/error_attribution.py` (all six drugs) — `phase3b_error_covariate_regression.csv`, `phase3b_error_shap.csv`, six PNGs
- **PASS** — `pytest api/tests/test_xai_real_backgrounds.py api/tests/test_feedback_loop.py api/tests/test_clinical_rules.py -v` — 7 passed
- **PASS** — `pytest api/tests/test_main.py` — 8 passed (schema change for `clinical_reasoning` is backward-compatible)

Note: A full `pytest api/tests` run may take several minutes if SHAP-heavy tests are included; the objective-specific suites above completed within normal CI time.
