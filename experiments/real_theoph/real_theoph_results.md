# Real Theoph Validation Results

**Experiment branch:** experiment/real-theoph-validation
**Checkpoint:** `artifacts/models/hybrid_gnn_pbpk_theophylline_demo_verify/model.pt`
**Inference mode:** `model.eval()` + `torch.no_grad()` — forward pass only, no retraining

## Covariate imputation (explicitly stated)

| Covariate | Source | Value used |
|-----------|--------|------------|
| `weight_kg` | R Theoph — real measured | per-subject (54.6–86.4 kg) |
| `dose_mg` | Computed: `dose_mgkg × weight_kg` | per-subject (211–320 mg) |
| `dose_mgkg` | R Theoph — real measured | per-subject (3.10–5.86 mg/kg) |
| `age_years` | **NOT in R Theoph — imputed** | 43.34 yr (training-set mean) |
| `sex` | **NOT in R Theoph — imputed** | 0.5437 for primary; 0/mean/1 for sensitivity |
| `f_bio` | Not in R Theoph | 1.0 (model default) |

SHAP importance reminder: `dose_mgkg`=1.608, `weight_kg`=1.060, `sex`=0.492, `age_years`=0.131.
Age imputation is low-risk; sex imputation introduces an uncertainty band quantified below.

## Pooled metrics — all subjects

| Scenario | R² | RMSE (mg/L) | n obs |
|----------|---:|------------:|------:|
| Naive baseline (predict mean) | +0.0000 | 2.8564 | 132 |
| **Model — all 132 observations** | **+0.6725** | **1.6346** | 132 |
| Model — excl. Subject-1 t=0 anomaly | +0.6675 | 1.6396 | 131 |

**Subject-1 t=0 note:** the 1-compartment ODE starts with zero drug in the system, so it
structurally predicts ≈0 mg/L at t=0. Subject 1 shows 0.74 mg/L at t=0 (all other subjects
show 0). In practice, excluding this single point barely moves either metric (ΔR²=−0.005,
ΔRMSE=+0.005 mg/L). The R² very slightly decreases rather than increases because removing
a low-concentration point shifts the mean of y_obs, which changes ss_tot more than it
changes ss_res for this small dataset. The effect is negligible and both figures are
reported for transparency.

**Naive-baseline check:** the naive baseline R² of +0.0000 (predicting the global mean
for every observation) confirms the model captures real structure beyond the mean.

## Per-subject results

| Subject | R² | RMSE mg/L | Sex-spread mg/L | CL L/h | Vd L | ka /h |
|--------:|---:|----------:|----------------:|-------:|-----:|------:|
|       1 |  +0.072 |     2.787 |        0.470 |    3.83 |    39.7 |   1.460 |
|       2 |  +0.855 |     1.101 |        0.447 |    3.38 |    34.6 |   1.495 |
|       3 |  +0.933 |     0.663 |        0.450 |    3.27 |    33.2 |   1.505 |
|       4 |  +0.857 |     1.055 |        0.452 |    3.40 |    34.7 |   1.495 |
|       5 |  +0.778 |     1.588 |        0.627 |    2.35 |    22.9 |   1.602 |
|       6 |  +0.898 |     0.663 |        0.471 |    3.85 |    39.9 |   1.458 |
|       7 |  +0.367 |     1.885 |        0.489 |    2.92 |    29.3 |   1.537 |
|       8 |  +0.902 |     0.734 |        0.448 |    3.27 |    33.2 |   1.505 |
|       9 |  +0.069 |     2.498 |        0.005 |    4.17 |    43.0 |   1.460 |
|      10 |  +0.448 |     2.161 |        0.571 |    2.55 |    25.1 |   1.577 |
|      11 |  +0.695 |     1.343 |        0.483 |    2.94 |    29.5 |   1.535 |
|      12 |  +0.842 |     1.327 |        0.540 |    2.68 |    26.6 |   1.562 |

Per-subject R²: min=0.069, median=0.810, max=0.933
Per-subject RMSE: min=0.663, median=1.335, max=2.787 mg/L

**Sex-spread** = max concentration difference across sex∈{0 (F), 0.544 (neutral), 1 (M)}
at any time point for that subject. Quantifies uncertainty from the unknown sex covariate.

## Interpretation

### What predicts well
The model reproduces the general shape of theophylline pharmacokinetics: rising absorption
phase followed by mono-exponential elimination. Subjects with weight and dose close to the
training distribution are expected to show higher per-subject R².

### What predicts poorly
- **Subject 1 t=0**: structurally unpredictable by the 1-cpt model (see note above).
- The real Theoph data spans 12 subjects with highly variable individual PK parameters;
  the model uses population-typical parameter estimates from simulated training data and
  cannot personalise to unmeasured individual covariates (e.g., CYP1A2 activity, smoking
  status, co-medications).

### Simulation-to-reality gap
The model was trained on **simulated** theophylline data; the simulated-test R² was 0.827.
The real-data R² of **0.673** (or **0.667** excl. t=0 anomaly) quantifies
the simulation-to-reality gap. A gap is the expected and informative result — it measures
how much of the model's predictive performance is driven by matching the simulator vs.
real biology. The model still captures substantial real-data structure (R² >> naive baseline
of 0.000).

## Files

| File | Description |
|------|-------------|
| `run_real_theoph_eval.py` | This evaluation script (forward pass only) |
| `real_theoph_results.md` | This report |
| `real_theoph_predictions.csv` | Per-observation predicted vs observed |
| `pred_vs_obs.png` | Scatter plot, all subjects |
| `concentration_curves.png` | 12-panel concentration-time profiles with sex uncertainty band |
