# Phase 3.3 — SHAP interpretation (patient covariates → predicted AUC)

KernelSHAP on the trained hybrid **per drug**, holding the molecular graph fixed. Background = random training patients. For each of **N** reference test patients, we explain the AUC response while holding that patient's dose, weight, F, and time grid fixed; only the z-scored patient tensor entries are coalition-perturbed. Approximate Shapley with ``nsamples=96``.

## Top 5 mean |SHAP| features per drug

### theophylline

| Rank | Feature | mean |SHAP| |
|---:|---|---:|
| 1 | dose_mgkg | 1.6078 |
| 2 | weight_kg | 1.0601 |
| 3 | sex | 0.4920 |
| 4 | age_years | 0.1314 |
| 5 | dose_mg | 0.0000 |

### warfarin

| Rank | Feature | mean |SHAP| |
|---:|---|---:|
| 1 | sex | 0.0192 |
| 2 | dose_mgkg | 0.0174 |
| 3 | weight_kg | 0.0164 |
| 4 | age_years | 0.0044 |
| 5 | dose_mg | 0.0000 |

### midazolam

| Rank | Feature | mean |SHAP| |
|---:|---|---:|
| 1 | sex | 0.0002 |
| 2 | age_years | 0.0000 |
| 3 | weight_kg | 0.0000 |
| 4 | dose_mgkg | 0.0000 |
| 5 | dose_mg | 0.0000 |

### caffeine

| Rank | Feature | mean |SHAP| |
|---:|---|---:|
| 1 | sex | 0.0103 |
| 2 | weight_kg | 0.0102 |
| 3 | dose_mgkg | 0.0083 |
| 4 | age_years | 0.0012 |
| 5 | dose_mg | 0.0000 |

### acetaminophen

| Rank | Feature | mean |SHAP| |
|---:|---|---:|
| 1 | log_dose_mg_per_kg | 0.0339 |
| 2 | weight_kg | 0.0248 |
| 3 | age_years | 0.0246 |
| 4 | sex | 0.0177 |
| 5 | dose_mg_per_kg | 0.0145 |

### digoxin

| Rank | Feature | mean |SHAP| |
|---:|---|---:|
| 1 | dose_mgkg | 0.0000 |
| 2 | sex | 0.0000 |
| 3 | age_years | 0.0000 |
| 4 | weight_kg | 0.0000 |
| 5 | dose_mg | 0.0000 |

## Pharmacological notes

- **weight_kg** (normalised channel): enters the ODE through absolute CL and V; even with dose fixed in mg, anthropometrics shift exposure per volume.
- **dose_mg** / **dose-normalised inputs**: primary driver of AUC for oral absorption.
- **age_years** / **sex**: captured here as coarse covariates in the fusion MLP head.

## Cross-drug patterns

Features appearing most often in this panel's top-5 (mean |SHAP|):

- **weight_kg** — in top-5 for 6 / 6 drugs.
- **sex** — in top-5 for 6 / 6 drugs.
- **age_years** — in top-5 for 6 / 6 drugs.
- **dose_mgkg** — in top-5 for 5 / 6 drugs.
- **dose_mg** — in top-5 for 5 / 6 drugs.
- **log_dose_mg_per_kg** — in top-5 for 1 / 6 drugs.
- **dose_mg_per_kg** — in top-5 for 1 / 6 drugs.

