# Final Research Report — Multi-Drug DL–PBPK Hybrid

**Project:** Structure-informed hybrid (GNN + mechanistic oral PK) trained on six drugs with benchmarks, uncertainty validation, and explainability.  
**Authoritative Phase 2 metrics:** `experiments/results/phase2_*_final.csv` (realistic PBPK-only baseline A1, σ=0.4 log-normal on CL and V per patient).  
**Data:** Simulated PK cohorts; single held-out split per drug (`SEED=42`).

---

## 1. Executive Summary

This work combines a **molecular graph neural network** with a **differentiable one-compartment oral PK simulator** to predict concentration–time profiles across six training drugs (theophylline, warfarin, midazolam, caffeine, acetaminophen, digoxin). The **DL–PBPK hybrid** achieves strong test $R^2$ on all six drugs (range **0.78–0.93** on the primary split) and **compares favourably to tabular ML and a vanilla curve GNN** while embedding physiological structure. A **corrected PBPK-only baseline** applies per-patient population uncertainty on clearance and volume (shared log-normal draw, σ=0.4), yielding a fair ablation ladder: mean test $R^2$ **0.341** (A1) vs **0.851** (full hybrid, A5).

**External stress test:** **Ibuprofen** zero-shot inference with a **frozen pretrained encoder** (no ibuprofen fine-tuning) yields test $R^2 \approx 0.84$ and RMSE $\approx 35\%$ of mean concentration—supporting cross-drug transfer of structure embeddings.

**Uncertainty:** Monte Carlo intervals ( $N{=}1000$, shared log-normal σ=0.3 on predicted CL and V) show **near-nominal pooled coverage** at the 90% level (~**0.875** empirical vs 0.90 nominal), but **per-drug calibration differs**: caffeine and acetaminophen are **under-covered** at nominal 90%, while theophylline and warfarin are slightly **over-covered**.

**Interpretability:** KernelSHAP on **predicted AUC** attributes variation primarily to **dose-normalised inputs**, **weight**, **sex**, and **age** for several drugs. **Midazolam** and **digoxin** show **near-flat patient-feature SHAP** under the **graph-fixed** explainer; this does **not** rank molecular importance—**graph-level attribution** would require a separate analysis (see §7 and §9).

**Disclosure:** All PK used for training and benchmarking is **simulated**; conclusions support **methods and software** claims more than **clinical deployment** without real-data validation.

---

## 2. Multi-drug validation results (six drugs × metrics)

Held-out **test** split per drug; $n=20$ test patients each. Source: `experiments/results/phase1_multidrug_metrics.csv` (DL–PBPK hybrid).

| Drug | $n_{\mathrm{test}}$ | Obs. mean (mg/L) | RMSE (mg/L) | RMSE % mean | MAE (mg/L) | MAPE (%) | $R^2$ | Cmax % err | AUC % err |
|------|--------------------:|-----------------:|------------:|------------:|-----------:|---------:|------:|-----------:|----------:|
| theophylline | 20 | 3.290 | 0.877 | 26.64 | 0.647 | 26.95 | 0.827 | 18.85 | 20.54 |
| warfarin | 20 | 0.482 | 0.118 | 24.49 | 0.086 | 21.81 | 0.781 | 17.31 | 15.54 |
| midazolam | 20 | 0.0118 | 0.00323 | 27.33 | 0.00194 | 6295.36† | 0.920 | 11.60 | 16.41 |
| caffeine | 20 | 2.071 | 0.445 | 21.48 | 0.317 | 20.93 | 0.871 | 15.05 | 12.80 |
| acetaminophen | 20 | 4.056 | 0.944 | 23.27 | 0.654 | 621209.28† | 0.929 | 13.32 | 17.93 |
| digoxin | 20 | 0.000262 | 7.72×10⁻⁵ | 29.48 | 5.71×10⁻⁵ | 24.40 | 0.780 | 23.98 | 20.91 |

† **MAPE** is unstable when observed concentrations are very small or cross near-zero numerical thresholds; rely on RMSE, $R^2$, and %-of-mean RMSE for low-concentration drugs.

---

## 3. Benchmark comparison (all baselines vs DL–PBPK)

Same splits and test cohorts as Phase 2. Source: `experiments/results/phase2_benchmark_metrics_final.csv`. **PBPK-only** uses the **realistic** population-uncertainty baseline (not literature-oracle CL/V).

### 3a. Test $R^2$

| Drug | PBPK-only | MLP | Random Forest | XGBoost | Vanilla GNN | **DL–PBPK** |
|------|----------:|----:|--------------:|--------:|--------------:|------------:|
| theophylline | −0.061 | 0.823 | 0.779 | 0.802 | 0.786 | **0.827** |
| warfarin | 0.347 | 0.777 | 0.810 | 0.763 | 0.819 | **0.781** |
| midazolam | 0.720 | 0.810 | 0.899 | 0.885 | 0.922 | **0.920** |
| caffeine | 0.305 | 0.860 | 0.853 | 0.817 | 0.868 | **0.871** |
| acetaminophen | 0.723 | 0.906 | 0.901 | 0.876 | 0.928 | **0.929** |
| digoxin | 0.014 | 0.367 | 0.637 | 0.644 | 0.569 | **0.780** |

### 3b. Test RMSE (mg/L)

| Drug | PBPK-only | MLP | Random Forest | XGBoost | Vanilla GNN | **DL–PBPK** |
|------|----------:|----:|--------------:|--------:|--------------:|------------:|
| theophylline | 2.173 | 0.887 | 0.992 | 0.939 | 0.975 | **0.877** |
| warfarin | 0.204 | 0.119 | 0.110 | 0.123 | 0.107 | **0.118** |
| midazolam | 0.00604 | 0.00497 | 0.00363 | 0.00387 | 0.00319 | **0.00323** |
| caffeine | 1.096 | 0.463 | 0.475 | 0.529 | 0.449 | **0.445** |
| acetaminophen | 1.923 | 1.083 | 1.114 | 1.248 | 0.952 | **0.944** |
| digoxin | 1.63×10⁻⁴ | 1.31×10⁻⁴ | 9.91×10⁻⁵ | 9.82×10⁻⁵ | 1.08×10⁻⁴ | **7.72×10⁻⁵** |

**Reading:** DL–PBPK is **best or tied-best** on $R^2$ for theophylline, caffeine, acetaminophen, and digoxin; for **warfarin** and **midazolam**, the **vanilla GNN** slightly edges $R^2$ while the hybrid remains competitive and preserves **interpretable PK parameters** and **ODE consistency**. PBPK-only with realistic uncertainty is **weak on several drugs** (negative or near-zero $R^2$ for theophylline and digoxin), motivating the hybrid.

---

## 4. Ablation study results (authoritative corrected summary)

Source: `experiments/results/phase2_ablation_summary_final.csv`. Means are averaged over the **six training drugs**.

| Variant | Description (short) | Mean test $R^2$ (6 drugs) | Mean test RMSE (6 drugs) |
|---------|---------------------|---------------------------:|-------------------------:|
| **A1** | Realistic PBPK-only (population σ=0.4 on CL & V) | **0.341** | 0.900 |
| **A2** | GNN-only (no mechanistic head coupling as in full hybrid setup) | 0.815 | 0.415 |
| **A3** | Hybrid without transfer init | 0.797 | 0.514 |
| **A4** | Hybrid, encoder frozen | 0.823 | 0.488 |
| **A5** | Full DL–PBPK | **0.851** | 0.398 |

**Important:** The **oracle** PBPK-only benchmark (literature CL/V matching the simulator ground truth) previously inflated A1 to ~**0.853**; that comparison was **invalid** for scientific claims. The **approved** staircase uses **A1 = 0.341** vs **A5 = 0.851**, demonstrating added value of the deep hybrid under **honest** classical uncertainty.

---

## 5. External validation (ibuprofen zero-shot)

Source: `experiments/results/phase2_external_validation.csv`.

| Drug | Split | $n$ | RMSE (mg/L) | RMSE % mean | MAE | $R^2$ | Cmax % err | AUC % err | Encoder | Fine-tuned on ibuprofen |
|------|-------|----:|------------:|------------:|----:|------:|-----------:|----------|---------|-------------------------|
| ibuprofen | test | 20 | 4.403 | 34.99 | 3.098 | 0.836 | 21.57 | 28.34 | pretrained_frozen | False |

The encoder was **not** fine-tuned on ibuprofen; this is a **zero-shot** structural transfer test. Metrics indicate **usable generalisation** for the synthetic external drug under the study’s data-generating assumptions.

---

## 6. Uncertainty calibration summary

**Procedure:** Monte Carlo $N=1000$; multiplicative log-normal uncertainty on hybrid-predicted **CL** and **V** (shared draw per sample, log-scale σ=**0.3**); central prediction intervals vs empirical coverage on held-out test concentration–time points. Source CSV: `experiments/results/phase3_uncertainty_calibration.csv`. **Nominal 0.90** is not a grid point; values below use **linear interpolation** between nominal **0.885** and **0.92** (see `experiments/reports/phase_3_calibration_shap_diagnostic.md`).

### 6a. Pooled (all six drugs)

| Scope | Nominal interval | Empirical coverage | Delta (empirical − nominal) |
|-------|-----------------:|-------------------:|-----------------------------:|
| ALL_DRUGS_POOLED | 0.90 | **0.875** | **−0.025** |

_pooled_ aggregates **1560** concentration observations (260 per drug × 6).

### 6b. Per-drug (nominal 0.90, interpolated)

| Drug | Nominal | Empirical coverage | Delta |
|------|--------:|-------------------:|------:|
| theophylline | 0.90 | 0.9077 | +0.0077 |
| warfarin | 0.90 | 0.9143 | +0.0143 |
| midazolam | 0.90 | 0.8896 | −0.0104 |
| caffeine | 0.90 | 0.8236 | −0.0764 |
| acetaminophen | 0.90 | 0.8637 | −0.0363 |
| digoxin | 0.90 | 0.8890 | −0.0110 |

**Message for the manuscript:** **Pooled** calibration can look acceptable while **per-drug** intervals are miscalibrated—especially **under-coverage for caffeine** (and moderately for acetaminophen) at nominal 90%. Drug-specific or hierarchical uncertainty models are a natural extension.

---

## 7. SHAP interpretation highlights

**Setup (Phase 3.3):** KernelSHAP on **predicted AUC**; **molecular graph fixed** per drug; only **z-scored patient tensor** entries are coalition-perturbed; `nsamples=96`; reference test patients per drug. Output: `experiments/results/phase3_shap_interpretation.md`, figure `experiments/plots/shap_summary_multidrug.png` (when regenerated).

**Cross-drug patterns:** **weight_kg**, **sex**, and **age_years** appear in the top-five mean |SHAP| lists for **all six** drugs; **dose_mgkg** (and related dose-normalised channels) dominate **theophylline** and rank highly for **acetaminophen**, **caffeine**, and **warfarin**.

**Midazolam and digoxin caveat:** Mean |SHAP| for **patient** features (sex, age, weight, dose) is **near zero** for these drugs in the published table. This reflects the **chosen explainer**, not a proof that covariates are globally irrelevant: with the **graph fixed**, AUC may be **locally flat** in patient-feature space at the reference points, or variation may fall below **sampling noise**.

**Graph-level attribution note:** This pipeline **does not compute SHAP values for molecular / GNN inputs**. Statements about **structure vs patient drivers** require **additional** methods (e.g. graph perturbation, GraphSHAP, or structural ablations comparing compounds). The Phase 3.3 narrative is strictly **patient-conditional** explanations of AUC.

---

## 8. Paper submission recommendation

**Priority order (with one-sentence rationale each):**

1. **Journal of Cheminformatics** — Primary target: strong fit for **open, reproducible cheminformatics** workflows combining molecular representations (GNN) with documented simulation backends and public-style methodology.  
2. **CPT: Pharmacometrics & Systems Pharmacology** — Domain-specific outlet for **hybrid mechanism–learning PK models**, uncertainty calibration dialogue, and translational framing for quantitative clinical pharmacology readers.  
3. **Briefings in Bioinformatics** — Broader **AI in life sciences** readership suitable if the narrative emphasises **benchmark rigour, ablations, and explainability positioning** alongside the PK application.

---

## 9. Known limitations & honest disclosures

- **Simulated vs real clinical data:** All training, test, and external ibuprofen evaluations use **synthetic** concentration–time data from the project’s generators. Results support **method comparison and software reproducibility**, not validated clinical performance.  
- **Single uncertainty parameter:** Monte Carlo calibration uses **one** shared log-normal σ on CL and V for all drugs and points; **no drug-specific** residual calibration, correlation between parameters, or observation noise model is fit—**per-drug miscalibration** (§6) is expected under this simplification.  
- **Patient-feature-only SHAP scope:** Explainability is **conditional on fixed structure**; **no** molecular Shapley values are reported; flat patient SHAP for some drugs **must not** be over-interpreted as evidence that chemistry is unimportant.  
- **Single train/validation/test split per drug:** Splits are fixed with **SEED=42**; metrics do **not** include cross-seed or nested cross-validation uncertainty; optimism bias is possible for model selection steps that touched the same split design.  
- **MAPE reliability:** For very low concentrations (e.g. digoxin, midazolam scales), **MAPE** can be numerically extreme; primary emphasis should remain on **RMSE**, **$R^2$**, and **RMSE as % of mean**.

---

## Artifact index (quick reference)

| Content | Path |
|---------|------|
| Phase 1 metrics | `experiments/results/phase1_multidrug_metrics.csv` |
| Phase 2 benchmarks (final) | `experiments/results/phase2_benchmark_metrics_final.csv` |
| Phase 2 ablations (final) | `experiments/results/phase2_ablation_summary_final.csv` |
| External ibuprofen | `experiments/results/phase2_external_validation.csv` |
| MC calibration | `experiments/results/phase3_uncertainty_calibration.csv` |
| MC + SHAP diagnostic | `experiments/reports/phase_3_calibration_shap_diagnostic.md` |
| SHAP narrative | `experiments/results/phase3_shap_interpretation.md` |
| Math / skeleton / checklist | `paper/mathematical_formulation.md`, `paper/paper_skeleton.md`, `paper/reproducibility_checklist.md` |

---

*Report generated as the consolidated final deliverable for Phases 1–4.*
