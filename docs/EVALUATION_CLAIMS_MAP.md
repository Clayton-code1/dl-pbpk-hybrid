# Evaluation ↔ API claims (honest limits)

This table ties **what the deployed API can claim** to **what the training/evaluation stack actually demonstrates**. Use it for thesis/examiner Q&A.

| Claim | Supported by | Limit / caveat |
|--------|----------------|----------------|
| Multi-compound routing | Per-drug dirs `hybrid_gnn_pbpk_{slug}_v1` + graphs | Only the **fixed six-drug panel**; novel compounds fall back to legacy theophylline GNN (SMILES) or MLP, not a universal drug model. |
| Covariates match training | `multidrug_utils.patient_feature_columns` mirrored in API + per-drug scalers | Default `age_years=40`, `sex=0` if omitted; mis-specified demographics shift predictions. |
| Oral F in simulation | `reference_pk` + dose scaling before PBPK-lite | F is **literature scalar**, not fitted per patient; IV regimens skip F scaling on the absorbed amount. |
| PK parameters from hybrid | Forward pass of trained hybrid | Reported CL/V/ka are **model outputs**, not independent measurements. |
| Safety / risk score | `risk_service` on Cmax & AUC vs refs | Thresholds are **tunable** calibration; not regulatory toxicity limits. |
| SHAP in `/explain/v2` | KernelSHAP (or finite-diff fallback) on risk_score | Panel path uses **synthetic local background** around the reference row—informative but not “population-grounded” like the theophylline JSON background. |
| Population bands | Lognormal scatter on CL, V, ka | Omegas are **generic priors**, not posterior from trial fits; `/predict/population` needs **MLP or panel** base PK (GNN-only without those can fail). |
| Drug structure delta | GNN vs MLP path in `drug_structure_effect` | **Disabled** for panel route: each panel drug has a **fixed** graph embedding; no cross-baseline within that endpoint. |
| Robustness demo | `multidrug_pk_bootstrap_demo.py`, `shap_seed` | Bootstrap is **feature-space noise** on normalized inputs—explores local sensitivity, not trial residual error. |

**Bottom line:** the API faithfully **routes** paper-aligned models for the training panel and aligns XAI with that predictor, but **generalization to new drugs, new populations, or clinical decisions** requires external validation not implied by the service alone.
