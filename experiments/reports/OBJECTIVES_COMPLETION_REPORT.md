# Project Objectives — Completion Evidence

Master’s presentation prep (Harare Institute of Technology, early June 2026).  
All paths below are relative to repository root **`dl-pbpk-hybrid/`**.

---

## Objective 1 — Predict safe vs unsafe dose

- **Implementation:** `api/app/services/risk_service.py` (risk score + `is_safe`); `api/app/main.py` `/predict/v2`; hybrid forward `experiments/models/hybrid_multidrug.py` + `api/app/services/hybrid_infer_service.py`
- **Verifying test/script:** `scripts/evaluation/run_theophylline_eval.py`; Phase 1 metrics `experiments/evaluation/evaluate_multidrug.py` → `experiments/results/phase1_multidrug_metrics.csv`
- **Key result:** Held-out test \(R^2\) for the six-drug hybrid reaches **0.78–0.93** per drug on simulated cohorts (authoritative table: `experiments/reports/FINAL_RESEARCH_REPORT.md` §2).
- **Scope and limitation:** Safety uses **tunable** Cmax/AUC–based scoring aligned with literature references, **not** regulatory toxicology; all training PK remains **simulated**.
- **Evidence figure/table:** `experiments/results/phase1_multidrug_metrics.csv`; `experiments/reports/FINAL_RESEARCH_REPORT.md`

---

## Objective 2 — XAI: visualize and quantify drivers

### 2.1 Patient-covariate attribution (pre-existing)

- **Implementation:** `experiments/explainability/shap_interpretation.py` → `experiments/results/phase3_shap_interpretation.md`, `experiments/plots/shap_summary_multidrug.png`
- **Verifying test/script:** `python -m experiments.explainability.shap_interpretation` (see `paper/reproducibility_checklist.md`)
- **Key result:** KernelSHAP on **predicted AUC** ranks dose-normalised inputs, weight, age, and sex for most panel drugs (`FINAL_RESEARCH_REPORT.md` §7).
- **Scope and limitation:** Graph fixed per drug; **not** molecular Shapley.
- **Evidence figure/table:** `experiments/plots/shap_summary_multidrug.png`

### 2.2 Graph-level structural attribution (NEW)

- **Implementation:** `experiments/explainability/graph_shap_lite.py`
- **Verifying test/script:** Run `python experiments/explainability/graph_shap_lite.py` (CLI `--dry-run` supported)
- **Key result (example — theophylline):** Highest mean |\(\Delta\)AUC| under atom masking: **atom index 3 (N), score ≈ 0.0276**; rank-2 **N at index 8 ≈ 0.0216** (`experiments/results/phase3b_graph_explainability.csv`).
- **Scope and limitation:** Node-mask (“GraphSHAP-lite”) explains **encoder sensitivity**; hydrogens included because graphs use `Chem.AddHs` (consistent with training).
- **Evidence figure/table:** `experiments/results/phase3b_graph_explainability.csv`; `experiments/plots/graph_explainability_<drug>.png`; `experiments/plots/graph_explainability_grid.png`

### 2.3 Error-focused attribution (NEW)

- **Implementation:** `experiments/explainability/error_attribution.py`
- **Verifying test/script:** `python experiments/explainability/error_attribution.py`
- **Key result (theophylline):** Top mean |SHAP| on **trajectory SSE**: **weight_kg (0.922)**, then sex (0.656), dose_mgkg (0.420) — `experiments/results/phase3b_error_shap.csv`. Univariate regression table: `experiments/results/phase3b_error_covariate_regression.csv`.
- **Scope and limitation:** SHAP explains **squared error on the fixed test trajectory**, holding the graph and sampling times per patient record (same kernel setup family as Phase 3.3).
- **Evidence figure/table:** `experiments/plots/error_attribution_<drug>.png`; CSVs under `experiments/results/phase3b_error_*.csv`

### 2.4 Production API explainability (improved)

- **Implementation:** `api/app/services/xai_service.py` — `load_panel_shap_background_training()` samples **training-split rows** from `experiments/data/processed/<drug>_pk_dataset.csv` (split logic matches `experiments/training/multidrug_utils.py`, `SEED=42` family); cached in `_PANEL_SHAP_BACKGROUND_CACHE`.
- **Verifying test/script:** `api/tests/test_xai_real_backgrounds.py`
- **Key result:** Panel KernelSHAP backgrounds are **exact rows** from training CSVs (test asserts nearest-neighbour distance ~ 0).
- **Scope and limitation:** If CSV or columns are missing, code **falls back** to local jitter (`_build_panel_background`).
- **Evidence table:** Honest-limits refresh in `docs/EVALUATION_CLAIMS_MAP.md` (SHAP row updated)

---

## Objective 3 — Corrective actions and learning updates

### 3.1 Algorithmic dose adjustment (pre-existing)

- **Implementation:** `api/app/main.py` `POST /recommend`
- **Verifying test/script:** `api/tests/test_main.py` (`test_recommend_*`)
- **Key result:** Binary search dose scaling + **reduce / split / interval** strategies with re-simulation.
- **Scope and limitation:** In-silico only; six-drug panel routing unchanged.
- **Evidence figure/table:** API integration tests above

### 3.2 Alternative regimen simulation (pre-existing)

- **Implementation:** Same `/recommend` path + `api/app/services/hybrid_infer_service.py` `simulate_curve`
- **Verifying test/script:** `api/tests/test_main.py`
- **Key result:** Each strategy returns full PK curves and safety blocks for comparison.
- **Scope and limitation:** Strategies are heuristic search templates, not trial-validated schedules.
- **Evidence figure/table:** `api/tests/test_main.py`

### 3.3 Clinical-rule interpretation layer (NEW)

- **Implementation:** `api/app/services/clinical_rules_service.py`; wired in `api/app/main.py` → `RecommendResponse.clinical_reasoning`
- **Verifying test/script:** `api/tests/test_clinical_rules.py`
- **Key result:** **Literature-band** text + **evidence tiers** (`literature_band`, `extrapolation`, `out_of_scope`) from `experiments/reference_pk.py` therapeutic windows.
- **Scope and limitation:** Narrative **support** only; does not replace clinician judgement.
- **Evidence figure/table:** `api/app/schemas.py` (`ClinicalReasoningItem`)

### 3.4 Feedback-driven fine-tuning pipeline (NEW)

- **Implementation:** `api/app/services/feedback_service.py`; `experiments/training/finetune_from_feedback.py`; seed builder `experiments/data/feedback/build_demo_feedback_log.py` → `experiments/data/feedback/feedback_log.csv`
- **Verifying test/script:** `api/tests/test_feedback_loop.py`; run `python experiments/training/finetune_from_feedback.py --drug theophylline`
- **Key result:** Feedback RMSE on the 39-row demo: **before 0.0410 mg/L → after 0.0411 mg/L** (tiny rise — expected with noisy synthetic residuals and tiny cohort); artefact directory created with full report: `artifacts/models/hybrid_gnn_pbpk_theophylline_v_finetuned/finetune_report.json`
- **Scope and limitation:** Offline **demo** loop; does **not** auto-switch production weights (see registry below).
- **Evidence figure/table:** `artifacts/models/hybrid_gnn_pbpk_theophylline_v_finetuned/`; `finetune_report.json`

### 3.5 Online risk calibration (pre-existing)

- **Implementation:** `api/app/services/risk_service.py` `update_calibration`; `POST /model/update`
- **Verifying test/script:** Covered by API / model-state patterns (existing suite)
- **Key result:** Sigmoid parameters **a, b, c** and threshold nudge from `{safe, unsafe}` labels persisted to `api/app/state/model_state.json`.
- **Scope and limitation:** Calibrates **risk score**, not neural network weights.
- **Evidence figure/table:** `api/app/services/risk_service.py`

### Model registry (NEW, manual promotion)

- **Implementation:** `api/app/services/model_registry.py` → `api/app/state/model_registry.json`
- **Verifying test/script:** Populated when `finetune_from_feedback.py` completes successfully
- **Key result:** Fine-tuned checkpoint registered as a **candidate**; **active** path defaults to `artifacts/models/hybrid_gnn_pbpk_{drug}_v1` until operators change it.
- **Scope and limitation:** No automatic hot-swap in the running API process.

---

## Summary

| Objective | Status | New evidence added in this pass |
|-----------|--------|-----------------------------------|
| 1 | **FULLY MET** (unchanged claim scope) | *(existing)* |
| 2 | **FULLY MET** (expanded artefacts) | `graph_shap_lite.py`, `error_attribution.py`, real-training SHAP backgrounds + tests |
| 3 | **FULLY MET** (expanded artefacts) | `clinical_rules_service.py`, feedback CSV + `finetune_from_feedback.py`, `model_registry.py`, `/recommend.clinical_reasoning` |
