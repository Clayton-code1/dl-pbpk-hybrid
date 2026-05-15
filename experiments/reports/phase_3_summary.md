=====================================================
PHASE 3 SUMMARY REPORT — CLINICAL SAFETY & UNCERTAINTY VALIDATION
=====================================================

STATUS: COMPLETE

WHAT WAS DONE:
- **§3.1 Literature-backed safety thresholds:** Peak concentration classification uses `therapeutic_min_mg_L` / `therapeutic_max_mg_L` from `REFERENCE_PK_DATA` (`experiments/safety/literature_bounds.py`, `experiments/safety/safety_thresholds.py`). The API `assess_risk(..., drug=...)` enriches output with a literature window when a canonical drug name is supplied (`api/app/services/risk_service.py`).
- **§3.2 Monte Carlo uncertainty calibration:** N=**1000** draws per evaluation point; shared log-normal multiplicative uncertainty on CL and V with log-scale σ=**0.3** (~±30%); empirical interval coverage vs nominal levels for each training drug plus **ALL_DRUGS_POOLED** (`experiments/uncertainty/monte_carlo_calibration.py` → `experiments/results/phase3_uncertainty_calibration.csv`, calibration plot under `experiments/plots/uncertainty_calibration.png` / `.pdf` when generated).
- **§3.3 SHAP pharmacological interpretation:** KernelSHAP on predicted **AUC** for all six training drugs; top **five** features by mean |SHAP| per drug, narrative and cross-drug comparison (`experiments/explainability/shap_interpretation.py` → `experiments/results/phase3_shap_interpretation.md`, `experiments/plots/shap_summary_multidrug.png` / `.pdf` when generated).

DATASETS / FILES CREATED OR MODIFIED:
- `experiments/results/phase3_safety_thresholds.md` — tabulated therapeutic bands and citations.
- `experiments/results/phase3_uncertainty_calibration.csv` — per-drug and pooled nominal vs empirical coverage.
- `experiments/results/phase3_shap_interpretation.md` — ranked features, pharmacological notes, cross-drug counts.
- `experiments/plots/uncertainty_calibration.png` (and `.pdf` if saved).
- `experiments/plots/shap_summary_multidrug.png` (and `.pdf` if saved).
- `api/app/services/risk_service.py` — optional `drug` argument and `literature` / `literature_status` enrichment.

KEY RESULTS:
- **§3.1:** Six drugs have explicit mg/L therapeutic bands from the reference table (see `phase3_safety_thresholds.md`).
- **§3.2:** Pooled coverage tracks nominal levels on the calibration grid. Example pooled rows: nominal **0.85** → empirical **~0.831**; nominal **0.885** → **~0.862**; nominal **0.92** → **~0.893** (1560 concentration-time points pooled across drugs; see CSV). No exact **0.90** grid point; linear interpolation between **0.885** and **0.92** gives empirical **~0.875** at nominal **0.90**.
- **§3.3:** Dose-normalised inputs and **weight_kg** dominate for several drugs; **sex** and **age_years** appear in all six top-5 panels (KernelSHAP setup: fixed molecular graph, z-scored patient tensor perturbations, `nsamples=96`, eight reference test patients per drug — details in `phase3_shap_interpretation.md`).

LIMITATIONS OR CAVEATS:
- Therapeutic windows are **population literature bands**, not individual targets; clinical context and protein binding differ by patient.
- MC calibration uses a **single** PK uncertainty model (log-σ=0.3 on CL and V); drug-specific or correlated uncertainty is not modelled.
- KernelSHAP is **approximate**; midazolam and digoxin show near-flat |SHAP| in this setup, which may reflect saturation or feature scaling rather than absence of covariate effects.

NEXT PHASE PREVIEW:
- Phase 4 closed — see `phase_4_summary.md`, `paper/mathematical_formulation.md`, `paper/paper_skeleton.md`, `paper/reproducibility_checklist.md`; optional sensitivity on MC σ / SHAP `nsamples` during manuscript polish.

=====================================================
PHASE 3 CLOSED
=====================================================

=== PHASE 3 COMPLETE — AWAITING APPROVAL TO PROCEED ===
