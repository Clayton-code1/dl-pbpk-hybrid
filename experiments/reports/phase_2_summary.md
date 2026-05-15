=====================================================
PHASE 2 SUMMARY REPORT — BENCHMARK COMPARISONS, ABLATIONS, STATISTICS, EXTERNAL VALIDATION
=====================================================

STATUS: COMPLETE (authoritative metrics: `*_final.csv` — see note below)

WHAT WAS DONE:
- §2.1 Trained/evaluated baselines on the same 80/10/10 splits as Phase 1 (PBPK-only, MLP, RF, XGBoost, Vanilla GNN, DL-PBPK); molecular + patient features for tabular models; Y standardisation for stable sklearn/vanilla-GNN training.
- §2.2 Ran ablations A1–A5: A1/A2/A5 from benchmark / Phase 1; trained **A3** and **A4** per drug under `artifacts/models/phase2_ablation_{A3,A4}_{drug}/`.
- §2.3 Paired t-tests vs DL-PBPK; significance heatmap.
- §2.4 Ibuprofen zero-shot (frozen encoder, no ibuprofen fine-tuning).
- **PBPK-only (A1) correction:** the naive PBPK baseline used literature CL/Vd identical to the simulation oracle (unfair). The **approved** pipeline applies **per-patient population uncertainty**: shared multiplicative log-normal factor `exp(σ·z)` on both CL and Vd (σ=**0.4** on the log scale), one draw per patient, reproducible per drug. Authoritative Phase 2 tables are **`phase2_benchmark_metrics_final.csv`**, **`phase2_ablation_summary_final.csv`**, **`phase2_statistical_tests_final.csv`** (copied from `*_corrected`; uncorrected and `_corrected` files retained for audit).

DATASETS / FILES CREATED OR MODIFIED:
- **Authoritative (paper):** `experiments/results/phase2_benchmark_metrics_final.csv`, `phase2_ablation_summary_final.csv`, `phase2_statistical_tests_final.csv`
- Audit: `phase2_benchmark_metrics.csv`, `phase2_benchmark_metrics_corrected.csv`, `phase2_prediction_cache*.pkl`, `phase2_ablation_by_drug.csv`, `phase2_ablation_summary.csv`, `phase2_ablation_summary_corrected.csv`, `phase2_statistical_tests.csv`, `phase2_statistical_tests_corrected.csv`
- `experiments/results/phase2_external_validation.csv`
- Plots: `experiments/plots/` (baseline bar, ablation, significance originals + `*_corrected` heatmap), ibuprofen scatter.
- `artifacts/models/phase2_ablation_A3_{drug}/`, `phase2_ablation_A4_{drug}/`

KEY RESULTS (corrected / final):
- **Ablation mean test R² (6 drugs):** A1 realistic PBPK **0.341**; A2 0.815; A3 0.797; A4 0.823; A5 **0.851** — staircase **A5 > A1** with population-parameter uncertainty on the classical baseline.
- **External (ibuprofen):** see `phase2_external_validation.csv`.

LIMITATIONS OR CAVEATS:
- Literature σ=0.4 on PK is a modelling choice for “clinician uncertainty”; sensitivity analyses can vary σ.
- If PK CSVs are regenerated, retrain hybrids and refresh Phase 2 artifacts.

NEXT PHASE PREVIEW:
- Phase 4 — manuscript packaging / optional external validation (Phase 3 closed: `phase_3_summary.md`).

=====================================================
PHASE 2 CLOSED — AUTHORITATIVE FILES ARE `*_final.csv`
=====================================================

=== PHASE 2 COMPLETE — AWAITING APPROVAL TO PROCEED ===
