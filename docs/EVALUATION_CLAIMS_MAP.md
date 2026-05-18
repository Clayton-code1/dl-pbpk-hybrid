# Evaluation ↔ API claims (honest limits)

This table ties **what the deployed API can claim** to **what the training/evaluation stack actually demonstrates**. Use it for thesis/examiner Q&A.

| Claim | Supported by | Limit / caveat |
|--------|----------------|----------------|
| Multi-compound routing | Per-drug dirs `hybrid_gnn_pbpk_{slug}_v1` + graphs | Only the **fixed six-drug panel**; novel compounds fall back to legacy theophylline GNN (SMILES) or MLP, not a universal drug model. |
| Covariates match training | `multidrug_utils.patient_feature_columns` mirrored in API + per-drug scalers | Default `age_years=40`, `sex=0` if omitted; mis-specified demographics shift predictions. |
| Oral F in simulation | `reference_pk` + dose scaling before PBPK-lite | F is **literature scalar**, not fitted per patient; IV regimens skip F scaling on the absorbed amount. |
| PK parameters from hybrid | Forward pass of trained hybrid | Reported CL/V/ka are **model outputs**, not independent measurements. |
| Safety / risk score | `risk_service` on Cmax & AUC vs refs | Thresholds are **tunable** calibration; not regulatory toxicity limits. |
| SHAP in `/explain/v2` | KernelSHAP (or finite-diff fallback) on risk_score | **Panel path:** `xai_service.load_panel_shap_background_training` samples **real training-split rows** from each `*_pk_dataset.csv` (SEED logic matches `multidrug_utils`); if the CSV is missing, the code falls back to **local jitter** around the reference row. Legacy MLP path still uses Theophylline JSON / synthetic backgrounds. |
| Population bands | Lognormal scatter on CL, V, ka | Omegas are **generic priors**, not posterior from trial fits; `/predict/population` needs **MLP or panel** base PK (GNN-only without those can fail). |
| Drug structure delta | GNN vs MLP path in `drug_structure_effect` | **Disabled** for panel route: each panel drug has a **fixed** graph embedding; no cross-baseline within that endpoint. |
| Robustness demo | `multidrug_pk_bootstrap_demo.py`, `shap_seed` | Bootstrap is **feature-space noise** on normalized inputs—explores local sensitivity, not trial residual error. |

**Bottom line:** the API faithfully **routes** paper-aligned models for the training panel and aligns XAI with that predictor, but **generalization to new drugs, new populations, or clinical decisions** requires external validation not implied by the service alone.

---

## Objective 3 — Corrective actions and learning (thesis mapping)

| Sub-claim | Implementation | Test / verification | Scope & limitation |
|-----------|------------------|---------------------|----------------------|
| Algorithmic dose reduction / regimen alternatives | `api/app/main.py` `POST /recommend` (binary search + strategies) | `api/tests/test_main.py` (`test_recommend_*`) | Deterministic **simulation** under the loaded hybrid; not a validated clinical titration protocol. |
| Literature-anchored clinical interpretation | `api/app/services/clinical_rules_service.py` | `api/tests/test_clinical_rules.py` | Uses **`experiments.reference_pk` therapeutic bands** for narrative tiering; **not** regulatory dose guidance. |
| Feedback logging | `api/app/services/feedback_service.py` (`record_feedback`, `summarize_feedback`) | `api/tests/test_feedback_loop.py` | CSV append-only audit trail; **no PHI** assumed—demo/synthetic rows only by default. |
| Feedback-driven fine-tuning (offline) | `experiments/training/finetune_from_feedback.py` | `api/tests/test_feedback_loop.py` (insufficient-data guard) + run produces `finetune_report.json` | **Small** Adam fine-tune on logged points; does **not** replace full Phase-1 retraining; **does not auto-promote** weights in the API. |
| Model registry (manual promotion) | `api/app/services/model_registry.py` | Exercised when fine-tune completes (candidate registration) | **Operators** must manually point serving code at new dirs if they choose to deploy. |
| Online risk-calibration nudges | `api/app/services/risk_service.py` `update_calibration` | Covered via `/model/update` integration patterns | Adjusts **sigmoid risk scoring**, not neural weights. |

### Objective 2 — New study artefacts (add-on)

| Sub-claim | Implementation | Verification |
|-----------|----------------|--------------|
| Graph-level attribution | `experiments/explainability/graph_shap_lite.py` | CSV + PNG under `experiments/results/phase3b_graph_explainability.csv`, `experiments/plots/graph_explainability_*.png` |
| Error-focused attribution | `experiments/explainability/error_attribution.py` | `phase3b_error_covariate_regression.csv`, `phase3b_error_shap.csv`, plots |
