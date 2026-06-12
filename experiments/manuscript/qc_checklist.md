# QC Checklist — Pre-Submission Verification

This checklist verifies that every quantitative claim in the manuscript traces to a specific, readable file in the repository. Tick each item before upload.

---

## A. Architecture Claims

| Claim | Source file | Line(s) | Status |
|-------|------------|---------|--------|
| Production model: `MultiDrugHybridGNNPBPK` | `experiments/models/hybrid_multidrug.py` | — | ✓ VERIFIED |
| Three outputs (CL_per_kg, Vd_per_kg, ka) | `experiments/models/hybrid_multidrug.py` | L95 `nn.Linear(head_hidden, 3)` | ✓ VERIFIED |
| V derived as `Vd_per_kg × weight_kg` | `experiments/models/hybrid_multidrug.py` | L143–153 | ✓ VERIFIED |
| GNN: 2 layers, node_feat=27, edge_feat=6, hidden=64, embed=64 | `artifacts/models/gnn_pretrain_combined_v1/config.json` | — | ✓ VERIFIED |
| Mean+max pooling → Linear(128,64) | `src/models/gnn/molecule_gnn.py` | L122–126 | ✓ VERIFIED |
| Total params: 94,403 (GNN=85,568; head=8,835) | Computed via venv Python instantiation | PowerShell count | ✓ VERIFIED |
| ODE: Euler 384 steps in training | `experiments/training/train_multidrug_hybrid.py` | L91 `N_EULER_STEPS = 384` | ✓ VERIFIED |
| 1-cpt ODE equations as stated | `src/models/ode/pk_1cpt_torch.py` | L1–8 | ✓ VERIFIED |
| Patient features: weight_kg, dose_mg, dose_mgkg, age_years, sex | `experiments/training/multidrug_utils.py` | L39–42 | ✓ VERIFIED |

---

## B. Data Generation Claims

| Claim | Source file | Status |
|-------|------------|--------|
| 200 virtual patients/drug | `experiments/data/download_pk_data.py` L67 | ✓ VERIFIED |
| 13 time points as listed | `experiments/data/download_pk_data.py` L87 | ✓ VERIFIED |
| 80/10/10 split, SEED=42 | `experiments/config.py` L20; `multidrug_utils.py` L166–187 | ✓ VERIFIED |
| Drug panel (6 training + ibuprofen external) | `experiments/config.py` DRUGS list | ✓ VERIFIED |
| IIV CV values (drug-specific) | `experiments/data/download_pk_data.py` L68 + drug-specific lines | ✓ VERIFIED |
| Noise fraction values | `experiments/data/download_pk_data.py` L77 + drug-specific | ✓ VERIFIED |
| Literature PK references per drug | `experiments/reference_pk.py` | ✓ VERIFIED |

---

## C. Pretraining Claims

| Claim | Source |
|-------|--------|
| Stage 2: 5,304 molecules (4,509 train / 795 val) | `artifacts/models/gnn_pretrain_combined_v1/config.json` |
| Stage 1: unsupervised masked node reconstruction, mask_rate=15% | Code: pretraining scripts |
| Datasets: Delaney + Lipophilicity + ChEMBL | config.json / pretraining code |
| CPU training | config.json |
| `transfer_initialisation=true` | config.json |

---

## D. Simulated Benchmark Results (Table 1)

All values are from `experiments/results/phase2_benchmark_metrics_final.csv`.
CSV column used: `R2`, `RMSE`, `RMSE_pct_of_mean`.
MAPE column **excluded** from manuscript (instability at near-zero concentrations). ✓

| Drug | DL-PK R² in CSV | DL-PK R² in manuscript | Match? |
|------|:-----------:|:-------------------:|:------:|
| theophylline | 0.8273736523931894 | 0.827 | ✓ |
| warfarin | 0.7811126975493047 | 0.781 | ✓ |
| midazolam | 0.9199409009893096 | 0.920 | ✓ |
| caffeine | 0.8707756516276923 | 0.871 | ✓ |
| acetaminophen | 0.9288735703357398 | 0.929 | ✓ |
| digoxin | 0.7796332173646252 | 0.780 | ✓ |
| **Mean** | **(0.8513 from ablation CSV)** | **0.851** | **✓** |

| Drug | PBPK-only R² in CSV | PBPK-only R² in manuscript | Match? |
|------|:---------:|:-----:|:------:|
| theophylline | -0.06143047018174275 | −0.061 | ✓ |
| warfarin | 0.3474417564084259 | 0.347 | ✓ |
| midazolam | 0.7198707733616045 | 0.720 | ✓ |
| caffeine | 0.3051512980928728 | 0.305 | ✓ |
| acetaminophen | 0.7226314526880315 | 0.723 | ✓ |
| digoxin | 0.014272591800598922 | 0.014 | ✓ |

---

## E. Ablation Results (Table 2)

All values from `experiments/results/phase2_ablation_summary_final.csv`.

| Condition | Mean R² in CSV | Manuscript | Match? |
|-----------|:-----------:|:----------:|:------:|
| A1 PBPK-only | 0.3413229003616318 | 0.341 | ✓ |
| A2 GNN-only | 0.8154576327754071 | 0.815 | ✓ |
| A3 hybrid_no_transfer | 0.7966992198533723 | 0.797 | ✓ |
| A4 hybrid_encoder_frozen | 0.8231928596903333 | 0.823 | ✓ |
| A5 Full_DLPBPK | 0.8512849483766435 | 0.851 | ✓ |

---

## F. External Validation — Ibuprofen

Source: `experiments/results/phase2_external_validation.csv`

| Claim | CSV value | Manuscript | Match? |
|-------|:------:|:---------:|:------:|
| R² | 0.8363323328511169 | 0.836 | ✓ |
| RMSE (mg/L) | 4.402677059173584 | 4.40 | ✓ |
| RMSE_pct | 34.992308052635366 | 35.0% | ✓ |
| n test patients | 20 | 20 | ✓ |
| encoder frozen, no fine-tune | fine_tuned_on_ibuprofen=False | ✓ stated | ✓ |

---

## G. Statistical Tests

Source: `experiments/results/phase2_statistical_tests_final.csv`

| Comparison | p-value in CSV | Reported | Match? |
|------------|:-------:|:--------:|:------:|
| Hybrid vs PBPK-only — theophylline | 0.0077 | p=0.0077 (**) | ✓ |
| Hybrid vs PBPK-only — warfarin | 0.0641 | p=0.064 (ns) | ✓ |
| Hybrid vs PBPK-only — midazolam | 0.0412 | p=0.041 (*) | ✓ |
| Hybrid vs PBPK-only — caffeine | 9.3×10⁻⁵ | p=9.3×10⁻⁵ (***) | ✓ |
| Hybrid vs PBPK-only — acetaminophen | 0.0031 | p=0.0031 (**) | ✓ |
| Hybrid vs PBPK-only — digoxin | 0.0179 | p=0.018 (*) | ✓ |
| Hybrid vs MLP — midazolam | 0.00041 | p=4.1×10⁻⁴ (***) | ✓ |
| Hybrid vs MLP — acetaminophen | 0.0419 | p=0.042 (*) | ✓ |

---

## H. Uncertainty Quantification

Source: `experiments/results/phase3_uncertainty_calibration.csv`

| Claim | CSV / calculation | Match? |
|-------|:---------:|:------:|
| Pooled 90% coverage ≈ 0.875 | Interpolated from CSV grid (0.885→0.8615, 0.92→0.8929) | ✓ |
| Per-drug at 90%: theoph=0.908 | CSV | ✓ |
| Per-drug at 90%: warfarin=0.914 | CSV | ✓ |
| Per-drug at 90%: midazolam=0.890 | CSV | ✓ |
| Per-drug at 90%: caffeine=0.824 | CSV | ✓ |
| Per-drug at 90%: acetaminophen=0.864 | CSV | ✓ |
| Per-drug at 90%: digoxin=0.889 | CSV | ✓ |

---

## I. Real-Data Validation (Table 3)

Sources: `experiments/real_theoph/real_theoph_results.md`, `experiments/warfarin_validation/warfarin_results.md`

| Claim | Source value | Manuscript | Match? |
|-------|:-------:|:---------:|:------:|
| Theoph: 12 subjects, 132 obs | MD file | 12 subj, 132 obs | ✓ |
| Theoph: R²=0.6725 | MD file | 0.673 | ✓ |
| Theoph: RMSE=1.6346 mg/L | MD file | 1.635 | ✓ |
| Theoph: naive R²=0.000, RMSE=2.856 | MD file | 0.000 / 2.856 | ✓ |
| Theoph: sim-test R²=0.827 | MD file | 0.827 | ✓ |
| Theoph: per-subject R² min=0.069, median=0.810, max=0.933 | MD file | ✓ stated | ✓ |
| Theoph: age imputed at 43.34 yr | MD file | ✓ stated | ✓ |
| Theoph: sex imputed at 0.544 | MD file | ✓ stated | ✓ |
| Warfarin: 32 subjects, 283 obs | MD file | 32 subj, 283 obs | ✓ |
| Warfarin all-32: R²=0.6945, RMSE=2.2778 | MD file | 0.695 / 2.278 | ✓ |
| Warfarin absorption n=13, 150 obs: R²=0.6681, RMSE=2.6966 | MD file | 0.668 / 2.697 | ✓ |
| Warfarin trough n=19, 133 obs: R²=0.7089, RMSE=1.685 | MD file | 0.709 / 1.685 | ✓ |
| Warfarin: no covariate imputation | MD file | ✓ stated | ✓ |
| Warfarin: dose extrapolation ~70–95 SD | MD file | ✓ stated | ✓ |
| Warfarin: no lag-time parameter | MD file | ✓ stated | ✓ |
| Warfarin: sim-test R²=0.781 | MD file | 0.781 | ✓ |

---

## J. Mandatory Exclusions — Confirmed Absent

| Excluded item | Reason | Status |
|---------------|--------|--------|
| MAPE metric | Computationally unstable near zero; column still in CSV but not reported | ✓ ABSENT from manuscript |
| "PBPK" as mechanistic term | This is a 1-cpt simulator, not physiologically-based | ✓ ABSENT — only "one-compartment" used |
| Therapeutic-window ML results (R² 0.17/0.30/0.23 etc.) | `experiments/level3_signal/` does NOT exist; results not reproducible | ✓ ABSENT |
| `HybridGNNPBPK` (old prototype) description | Predicts only CL+ka; V supplied externally; NOT the production model | ✓ ABSENT |
| Prototype in Results / Methods | Only `MultiDrugHybridGNNPBPK` described | ✓ ABSENT |

---

## K. Items Requiring Author Action Before Upload

- [ ] Confirm ibuprofen PK reference (Greenblatt & Koch-Weser 1975) is correct
- [ ] Confirm warfarin dataset preferred citation (O'Reilly original / Holford 1986 / nlmixr2data package)
- [ ] Supply ORCID for both authors
- [ ] Confirm departmental affiliation line (e.g., "Department of Biomedical Engineering")
- [ ] Confirm corresponding author email (institutional preferred?)
- [ ] Confirm whether all three ADME pretraining datasets (Delaney, Lipophilicity, ChEMBL) should appear in Methods
- [ ] Decision on bioavailability F=1.0 limitation disclosure placement
- [ ] Generate Figure 2 (benchmark bar chart) and Figure 3 (ablation ladder) from results CSVs
- [ ] Decide whether figure files already on disk (real_theoph/pred_vs_obs.png, warfarin_validation/pred_vs_obs.png, concentration_curves.png) are final-quality for submission

---

*QC checklist last updated: 2026-06-09. All source files verified from repository `c:\Users\Admin\dl-pbpk-hybrid`.*
