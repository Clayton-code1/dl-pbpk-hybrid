# Phase 3 post-approval diagnostic (no retraining)

## 1. Per-drug MC calibration at nominal 90% interval

Source: `experiments/results/phase3_uncertainty_calibration.csv`. The CSV uses nominal grid points **0.885** and **0.919999…** (float representation of **0.92**); **empirical coverage at nominal 0.90** is obtained by **linear interpolation** between those two rows per drug.

| Drug | Nominal | Empirical coverage | Delta (empirical − nominal) |
|------|--------:|-------------------:|------------------------------:|
| theophylline | 0.90 | 0.9077 | +0.0077 |
| warfarin | 0.90 | 0.9143 | +0.0143 |
| midazolam | 0.90 | 0.8896 | −0.0104 |
| caffeine | 0.90 | 0.8236 | −0.0764 |
| acetaminophen | 0.90 | 0.8637 | −0.0363 |
| digoxin | 0.90 | 0.8890 | −0.0110 |

**Interpretation:** Pooled coverage near 0.90 masks **heterogeneity**: theophylline and warfarin are slightly **over-covered** at nominal 90%; caffeine and acetaminophen are **under-covered** (largest gap for caffeine). Midazolam and digoxin are close to nominal. No model retraining is implied—this is a readout of the existing MC procedure ($N=1000$, log-σ=0.3 on CL and V).

## 2. SHAP: midazolam and digoxin (patient vs molecular)

Source: `experiments/results/phase3_shap_interpretation.md` and `experiments/explainability/shap_interpretation.py`.

- Phase 3.3 runs **KernelSHAP only on the z-scored patient feature vector**, with the **molecular graph held fixed** for each drug. **Molecular / GNN inputs are not assigned SHAP values in this pipeline.**
- For **midazolam** and **digoxin**, the reported near-zero mean |SHAP| values apply to **patient** channels (sex, age_years, weight_kg, dose_mgkg, dose_mg): the markdown tables list no molecular descriptors.

**Conclusion:** The flat patient-feature SHAP does **not** mean “structure is unimportant”; it means that **under this explainer** (graph fixed, patient vector perturbed), predicted AUC is **locally insensitive** to those patient-feature coalitions—or variance is below the numerical / `nsamples=96` resolution. Any statement about **relative** molecular vs patient importance would require a **separate** explanation that varies graph inputs (e.g. GraphSHAP or paired structural ablations), which was out of scope for the Phase 3.3 script.
