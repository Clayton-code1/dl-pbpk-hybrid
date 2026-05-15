# Paper skeleton — multi-drug DL–PBPK hybrid

Suggested article type: **Methods / modelling** with computational experiments (synthetic multi-drug PK + external drug transfer). Section titles are placeholders; tailor to target journal guidelines.

---

## Title

Deep graph–physiology hybrid for multi-drug oral pharmacokinetics: benchmarks, calibration, and safety-aware inference.

## Abstract (~200–250 words)

- **Background:** Limitations of purely data-driven vs purely mechanistic PK models across novel chemical matter.
- **Objective:** Joint molecular graph encoding + identifiable one-compartment ODE head; multi-drug training and zero-shot transfer.
- **Methods:** Dataset simulation; `MultiDrugHybridGNNPBPK`; baselines (PBPK-only, tabular ML, vanilla GNN); ablations; literature therapeutic bands; MC calibration; KernelSHAP on patient covariates.
- **Results:** Headline test metrics; realistic PBPK ablation staircase; external drug; calibration coverage highlights; SHAP patterns.
- **Conclusions:** When mechanistic structure helps; caveats (synthetic data, uncertainty model).

---

## 1. Introduction

- Clinical need for **structure-informed** PK early in drug development.
- Gaps in PBPK parameter identification vs black-box ML generalisation.
- Contributions (bullet list): (i) hybrid architecture + training protocol, (ii) rigorous baseline and ablation ladder including **fair** PBPK-only uncertainty, (iii) safety and uncertainty validation, (iv) explainability scope and limitations.

## 2. Related work

- PBPK and compartment models; ML in PK (non-compartmental, NN, GNN on SMILES/graphs).
- Hybrid physics-informed / grey-box models; transfer and multi-task learning in drug discovery.
- Uncertainty quantification and SHAP in healthcare ML (position KernelSHAP **conditional** explanations).

## 3. Methods

### 3.1 Data generation and drugs

- Six training drugs + one external drug; patients per drug; splits (e.g. 80/10/10); reproducible seeds (`experiments/config.py`).
- Simulation noise, IIV, and literature-based PK anchors (`download_pk_data`, `reference_pk`).

### 3.2 Molecular featurisation

- RDKit / graph construction; node and edge dimensions (align with `MoleculeGNN` config).

### 3.3 Hybrid model

- Equations from `mathematical_formulation.md`: GNN $\to$ MLP $\to$ $(\mathrm{CL}_{\mathrm{kg}}, \mathrm{V}_{\mathrm{kg}}, k_a)$; weight scaling; oral $F$; Euler simulator.

### 3.4 Training objective and schedules

- Loss (PK supervision weighting by drug if used); encoder freeze/unfreeze policy for difficult drugs (e.g. warfarin).

### 3.5 Baselines and ablations

- PBPK-only with **population uncertainty** (log-normal $\sigma=0.4$ on CL and V); MLP/RF/XGBoost; vanilla curve GNN; full hybrid (A1–A5 narrative).

### 3.6 Statistical comparison

- Paired tests vs hybrid; multiple testing caveat; report effect sizes and CIs where possible.

### 3.7 Safety and literature thresholds

- Therapeutic concentration bands from `REFERENCE_PK_DATA`; API enrichment (`risk_service`).

### 3.8 Uncertainty calibration

- MC scheme: $N=1000$, shared log-normal on $\mathrm{CL}, V$, $\sigma_{\mathrm{mc}}=0.3$; nominal vs empirical coverage (per drug + pooled).

### 3.9 Explainability

- KernelSHAP on predicted AUC; **graph fixed**; patient vector perturbed; `nsamples`, reference patients; limitation for molecular attribution.

## 4. Results

### 4.1 Predictive accuracy

- Per-drug test $R^2$, RMSE; table from `phase1_multidrug_metrics.csv` / Phase 2 benchmark CSVs (**final** authoritative tables).

### 4.2 Baselines and ablation staircase

- Corrected A1 vs A5; significance heatmap excerpt.

### 4.3 External validation

- Ibuprofen (or other) zero-shot metrics; figure: observed vs predicted.

### 4.4 Calibration

- Calibration plot + per-drug interpolated nominal 0.90 row (`phase_3_calibration_shap_diagnostic.md`).

### 4.5 SHAP

- Multi-panel summary figure; cross-drug feature patterns; flat patient SHAP for select drugs **in this explainer setup**.

## 5. Discussion

- Interpretation of hybrid inductive bias; when graph signal dominates.
- **Heterogeneous** MC calibration across drugs; caffeine/acetaminophen under-coverage at 90%.
- Explainability: patient-conditional SHAP ≠ global structure importance; future graph-level methods.
- Synthetic data limits; path to clinical cohorts and richer PBPK.

## 6. Conclusion

- Three bullet takeaways; reproducibility pointer.

## Declarations

- Code availability, data (synthetic generation scripts), ethics (if human data later).

## References

- Populate from `phase3_safety_thresholds.md` and standard PK / ML citations.

---

## Figure / table list (draft)

| ID | Content |
|----|--------|
| Fig. 1 | Model diagram (GNN + head + ODE). |
| Fig. 2 | Multi-drug predicted vs observed grid. |
| Fig. 3 | Baseline / ablation bar or staircase. |
| Fig. 4 | External drug validation. |
| Fig. 5 | Uncertainty calibration curve. |
| Fig. 6 | SHAP summary multi-drug. |
| Table 1 | Dataset and split summary. |
| Table 2 | Main test metrics + baselines. |
| Table 3 | Therapeutic windows (optional supplement). |
